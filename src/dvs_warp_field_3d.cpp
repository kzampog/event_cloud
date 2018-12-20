#include <cilantro/cilantro.hpp>

void warp_toggle(cilantro::Visualizer &viz) {
    viz.toggleVisibility("wf");
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

    cilantro::PointCloud3f frame1(argv[1]);
    cilantro::PointCloud3f frame2(argv[2]);


    Eigen::Vector3f mean = frame1.points.rowwise().mean();

    float minz = frame1.points.row(2).minCoeff();
    float maxz = frame1.points.row(2).maxCoeff();
    float zrange_to_keep = (maxz - minz)/1.0f;

    float min_keep = mean[2] - zrange_to_keep/2.0f;
    float max_keep = mean[2] + zrange_to_keep/2.0f;

    std::vector<size_t> remove1, remove2;
    for (size_t i = 0; i < frame1.size(); i++)
        if (frame1.points(2,i) < min_keep || frame1.points(2,i) > max_keep) remove1.emplace_back(i);
    for (size_t i = 0; i < frame2.size(); i++)
        if (frame2.points(2,i) < min_keep || frame2.points(2,i) > max_keep) remove2.emplace_back(i);

    frame1.remove(remove1);
    frame2.remove(remove2);



    frame1.points.colwise() -= mean;
    frame1.points.row(2).array() += 2.0f;
    frame2.points.colwise() -= mean;
    frame2.points.row(2).array() += 2.0f;

    frame1.estimateNormalsKNN(30);
//    frame2.estimateNormalsKNN(30);

    size_t iter;
    cilantro::Timer timer;
    timer.start();
//    auto wf = getDenseWarpField<cilantro::AffineTransform3f>(frame1, frame2, iter);
//    auto wf = getDenseWarpField<cilantro::RigidTransform3f>(frame1, frame2, iter);
//    auto wf = getSparseWarpField<cilantro::AffineTransform3f>(frame1, frame2, iter);
    auto wf = getSparseWarpField<cilantro::RigidTransform3f>(frame1, frame2, iter);
    timer.stop();
    std::cout << "Total registration time: " << timer.getElapsedTime() << "ms (" << iter << " iterations)" << std::endl;

    auto frame2_t = frame2.transformed(wf);



    pangolin::CreateWindowAndBind("test",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual)
            .AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("test", "disp1");
    cilantro::Visualizer viz2("test", "disp2");

    viz1.registerKeyboardCallback('w', std::bind(warp_toggle, std::ref(viz1)));

    viz1.addObject<cilantro::PointCloudRenderable>("frame1", frame1, cilantro::RenderingProperties().setPointColor(0,0,1).setUseLighting(false));
    viz1.addObject<cilantro::PointCloudRenderable>("frame2", frame2, cilantro::RenderingProperties().setPointColor(1,0,0).setUseLighting(false));
    viz1.addObject<cilantro::PointCorrespondencesRenderable>("wf", frame2_t, frame2, cilantro::RenderingProperties().setLineWidth(0.1f));

    viz2.addObject<cilantro::PointCloudRenderable>("frame1", frame1, cilantro::RenderingProperties().setPointColor(0,0,1).setUseLighting(false));
    viz2.addObject<cilantro::PointCloudRenderable>("frame2_t", frame2_t, cilantro::RenderingProperties().setPointColor(1,0,0).setUseLighting(false));

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
