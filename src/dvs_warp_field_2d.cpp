#include <cilantro/cilantro.hpp>

void warp_toggle(cilantro::Visualizer &viz) {
    viz.toggleVisibility("wf");
}

cilantro::PointCloud3f makeVisualizable(const cilantro::PointCloud2f &cloud) {
    cilantro::PointCloud3f res;
    res.points.resize(3, cloud.size());
    res.points.topRows(2) = cloud.points;
    res.points.row(2).setConstant(1.5f);
    if (cloud.hasNormals()) {
        res.normals.resize(3, cloud.size());
        res.normals.topRows(2) = cloud.normals;
        res.normals.row(2).setZero();
    }
    if (cloud.hasColors()) {
        res.colors = cloud.colors;
    }
    return res;
}

void readAllEvents(const std::string &fname, cilantro::VectorSet2f &events, std::vector<double> &ts) {
    Eigen::MatrixXd data;
    cilantro::readEigenMatrixFromFile(fname, data, false);

    float mu_x = data.col(1).mean();
    float mu_y = data.col(2).mean();

    events.resize(2, data.rows());
    events.row(0) = (data.col(1).cast<float>().array() - mu_x)/200.0f;
    events.row(1) = (data.col(2).cast<float>().array() - mu_y)/200.0f;

    double start_ts = data.col(0).minCoeff();
    ts.resize(data.rows());
    Eigen::Map<Eigen::VectorXd>(ts.data(), ts.size(), 1) = data.col(0).array() - start_ts;
}

int getIndexByTS(const std::vector<double> &ts, double val) {
    int low = 0;
    int high = ts.size() - 1;
    int mid;

    while (low <= high) {
        mid = low + (high - low)/2;
        if (val == ts[mid]) {
            return mid;
        } else if (val < ts[mid]) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }

    return mid;
}

cilantro::PointCloud2f getFrame(const cilantro::VectorSet2f &events, int start, int end) {
    cilantro::PointCloud2f res;
    res.points.resize(2, end - start + 1);
    for (int i = 0; i < res.points.cols(); i++) {
        res.points.col(i) = events.col(start + i);
    }
    return res;
}

template <class TransformT>
cilantro::TransformSet<TransformT> getSparseWarpField(const cilantro::PointCloud<typename TransformT::Scalar,TransformT::Dim> &frame1,
                                                      const cilantro::PointCloud<typename TransformT::Scalar,TransformT::Dim> &frame2,
                                                      size_t &iter)
{
    typedef typename TransformT::Scalar Scalar;
    enum { Dim = TransformT::Dim };

    Scalar control_res = 0.01;
    Scalar src_to_control_sigma = 0.02;
    Scalar regularization_sigma = 0.03;

    Scalar max_correspondence_dist_sq = 0.1*0.1;

    // Get a sparse set of control nodes by downsampling
    cilantro::VectorSet<Scalar,Dim> control_points = cilantro::PointsGridDownsampler<Scalar,Dim>(frame2.points, control_res).getDownsampledPoints();
    cilantro::KDTree<Scalar,Dim> control_tree(control_points);

    // Find which control nodes affect each point in src
    std::vector<cilantro::NeighborSet<Scalar>> src_to_control_nn;
    control_tree.search(frame2.points, cilantro::KNNNeighborhoodSpecification(3), src_to_control_nn);

    // Get regularization neighborhoods for control nodes
    std::vector<cilantro::NeighborSet<Scalar>> regularization_nn;
    control_tree.search(control_points, cilantro::KNNNeighborhoodSpecification(5), regularization_nn);

    cilantro::SimpleCombinedMetricSparseWarpFieldICP<TransformT> icp(frame1.points, frame1.normals, frame2.points, src_to_control_nn, control_points.cols(), regularization_nn);

    icp.correspondenceSearchEngine().setMaxDistance(max_correspondence_dist_sq);
    icp.controlWeightEvaluator().setSigma(src_to_control_sigma);
    icp.regularizationWeightEvaluator().setSigma(regularization_sigma);

    icp.setMaxNumberOfIterations(10).setConvergenceTolerance(2.5e-3);
    icp.setMaxNumberOfGaussNewtonIterations(1).setGaussNewtonConvergenceTolerance(5e-4);
    icp.setMaxNumberOfConjugateGradientIterations(500).setConjugateGradientConvergenceTolerance(1e-5);
    icp.setPointToPointMetricWeight(0.1).setPointToPlaneMetricWeight(1.0).setStiffnessRegularizationWeight(200.0);
    icp.setHuberLossBoundary(1e-2);

    icp.estimate();
    iter = icp.getNumberOfPerformedIterations();

    return icp.getDenseWarpField();
}

template <class TransformT>
cilantro::TransformSet<TransformT> getDenseWarpField(const cilantro::PointCloud<typename TransformT::Scalar,TransformT::Dim> &frame1,
                                                     const cilantro::PointCloud<typename TransformT::Scalar,TransformT::Dim> &frame2,
                                                     size_t &iter)
{
    typedef typename TransformT::Scalar Scalar;
    enum { Dim = TransformT::Dim };

    std::vector<cilantro::NeighborSet<Scalar>> regularization_nn;
    cilantro::KDTree<Scalar,Dim>(frame2.points).search(frame2.points, cilantro::KNNNeighborhoodSpecification(12), regularization_nn);

    cilantro::SimpleCombinedMetricDenseWarpFieldICP<TransformT> icp(frame1.points, frame1.normals, frame2.points, regularization_nn);

    Scalar max_correspondence_dist_sq = 0.1*0.1;
    Scalar regularization_sigma = 0.05;

    icp.correspondenceSearchEngine().setMaxDistance(max_correspondence_dist_sq);
    icp.regularizationWeightEvaluator().setSigma(regularization_sigma);

    icp.setMaxNumberOfIterations(15).setConvergenceTolerance(2.5e-4);
    icp.setMaxNumberOfGaussNewtonIterations(1).setGaussNewtonConvergenceTolerance(5e-4);
    icp.setMaxNumberOfConjugateGradientIterations(500).setConjugateGradientConvergenceTolerance(1e-5);
    icp.setPointToPointMetricWeight(0.1).setPointToPlaneMetricWeight(1.0).setStiffnessRegularizationWeight(200.0);
    icp.setHuberLossBoundary(1e-2);

    icp.estimate();
    iter = icp.getNumberOfPerformedIterations();

    return icp.getTransform();
}

int main(int argc, char ** argv) {
    const double frame_duration = 5e-3;
    const double t_skip = 30e-3;

    cilantro::VectorSet2f events;
    std::vector<double> ts;
    readAllEvents(argv[1], events, ts);

    double start = 40e-3;

    double start1 = start;
    double end1 = start1 + frame_duration;
    double start2 = end1 + t_skip;
    double end2 = start2 + frame_duration;

    cilantro::PointCloud2f frame1 = getFrame(events, getIndexByTS(ts, start1), getIndexByTS(ts, end1));
    cilantro::PointCloud2f frame2 = getFrame(events, getIndexByTS(ts, start2), getIndexByTS(ts, end2));

//    const float rad = 1.0f/200.0f + 10.0f*std::numeric_limits<float>::epsilon();
//    const float rad_sq = rad*rad;
//
//    std::vector<cilantro::NeighborSet<float>> nn1;
//    std::vector<cilantro::NeighborSet<float>> nn2;
//    cilantro::KDTree2f(frame1.points).radiusSearch(frame1.points, rad_sq, nn1);
//    cilantro::KDTree2f(frame2.points).radiusSearch(frame2.points, rad_sq, nn2);
//
//    std::vector<size_t> remove1, remove2;
//    for (size_t i = 0; i < frame1.size(); i++)
//        if (nn1[i].size() >= 5) remove1.emplace_back(i);
//    for (size_t i = 0; i < frame2.size(); i++)
//        if (nn2[i].size() >= 5) remove2.emplace_back(i);
//
//    frame1.remove(remove1);
//    frame2.remove(remove2);

    frame1.estimateNormalsKNN(20);
//    frame2.estimateNormalsKNN(20);



    size_t iter;
    cilantro::Timer timer;
    timer.start();
//    auto wf = getDenseWarpField<cilantro::AffineTransform2f>(frame1, frame2, iter);
    auto wf = getDenseWarpField<cilantro::RigidTransform2f>(frame1, frame2, iter);
//    auto wf = getSparseWarpField<cilantro::AffineTransform2f>(frame1, frame2, iter);
//    auto wf = getSparseWarpField<cilantro::RigidTransform2f>(frame1, frame2, iter);
    timer.stop();
    std::cout << "Total registration time: " << timer.getElapsedTime() << "ms (" << iter << " iterations)" << std::endl;

    auto frame2_t = frame2.transformed(wf);



    auto frame1_v = makeVisualizable(frame1);
    auto frame2_v = makeVisualizable(frame2);
    auto frame2_t_v = makeVisualizable(frame2_t);

    pangolin::CreateWindowAndBind("test",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
            .AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("test", "disp1");
    cilantro::Visualizer viz2("test", "disp2");

    viz1.registerKeyboardCallback('w', std::bind(warp_toggle, std::ref(viz1)));

    viz1.addObject<cilantro::PointCloudRenderable>("frame1", frame1_v, cilantro::RenderingProperties().setPointColor(0,0,1).setUseLighting(false));
    viz1.addObject<cilantro::PointCloudRenderable>("frame2", frame2_v, cilantro::RenderingProperties().setPointColor(1,0,0).setUseLighting(false));
    viz1.addObject<cilantro::PointCorrespondencesRenderable>("wf", frame2_t_v, frame2_v, cilantro::RenderingProperties().setLineWidth(0.1f));

    viz2.addObject<cilantro::PointCloudRenderable>("frame1", frame1_v, cilantro::RenderingProperties().setPointColor(0,0,1).setUseLighting(false));
    viz2.addObject<cilantro::PointCloudRenderable>("frame2_t", frame2_t_v, cilantro::RenderingProperties().setPointColor(1,0,0).setUseLighting(false));

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
