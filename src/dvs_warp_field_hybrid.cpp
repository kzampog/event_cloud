#include <cilantro/cilantro.hpp>

void warp_toggle(cilantro::Visualizer &viz) {
    viz.toggleVisibility("wf");
}

void readAllEvents3D(const std::string &fname, cilantro::VectorSet3f &events) {
    Eigen::MatrixXd data;
    cilantro::readEigenMatrixFromFile(fname, data, false);

    float mu_x = data.col(1).mean();
    float mu_y = data.col(2).mean();

    events.resize(3, data.rows());
    events.row(0) = (data.col(1).cast<float>().array() - mu_x)/200.0f;
    events.row(1) = (data.col(2).cast<float>().array() - mu_y)/200.0f;
    double start_ts = data.col(0).minCoeff();
    events.row(2) = (data.col(0).array() - start_ts).cast<float>();

}

template <typename TSVectorT, typename TST>
int getIndexByTS(const TSVectorT &ts, TST val) {
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

template <ptrdiff_t Dim>
cilantro::PointCloud<float,Dim> getFrame(const cilantro::VectorSet<float,Dim> &events, int start, int end) {
    cilantro::PointCloud<float,Dim> res;
    res.points.resize(Dim, end - start + 1);
    for (int i = 0; i < res.points.cols(); i++) {
        res.points.col(i) = events.col(start + i);
    }
    return res;
}

template <class TransformT>
void transformPointsHybrid(const cilantro::TransformSet<TransformT> &tforms,
                           const cilantro::ConstDataMatrixMap<typename TransformT::Scalar,TransformT::Dim + 1> &points,
                           cilantro::DataMatrixMap<typename TransformT::Scalar,TransformT::Dim + 1> result)
{
    typedef typename TransformT::Scalar Scalar;
    enum { Dim = TransformT::Dim };

#pragma omp parallel for
    for (size_t i = 0; i < points.cols(); i++) {
        Eigen::Matrix<Scalar,Dim + 1,Dim + 1> linear(Eigen::Matrix<Scalar,Dim + 1,Dim + 1>::Identity());
        linear.template block<Dim,Dim>(0,0) = tforms[i].linear();
        Eigen::Matrix<Scalar,Dim + 1,1> translation;
        translation.template segment<Dim>(0) = tforms[i].translation();
        translation(Dim) = 0;
        result.col(i).noalias() = linear*points.col(i) + translation;
    }
}

template <typename ScalarT, ptrdiff_t EigenDim>
class HybridPointsAdaptor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef ScalarT Scalar;

    enum { FeatureDimension = EigenDim };

    HybridPointsAdaptor(const cilantro::ConstVectorSetMatrixMap<ScalarT,EigenDim> &points)
            : data_map_(points),
              transformed_data_(points.rows(), points.cols()),
              transformed_data_map_(transformed_data_)
    {}

    HybridPointsAdaptor& transformFeatures() {
#pragma omp parallel for
        for (size_t i = 0; i < data_map_.cols(); i++) {
            transformed_data_.col(i) = data_map_.col(i);
        }
        return *this;
    }

    template <class TransformT>
    inline HybridPointsAdaptor& transformFeatures(const cilantro::TransformSet<TransformT> &tforms) {
        transformPointsHybrid(tforms, data_map_, transformed_data_);
        return *this;
    }

    inline const cilantro::ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getFeaturesMatrixMap() const {
        return data_map_;
    }

    inline const cilantro::ConstVectorSetMatrixMap<ScalarT,FeatureDimension>& getTransformedFeaturesMatrixMap() const {
        return transformed_data_map_;
    }

protected:
    cilantro::ConstVectorSetMatrixMap<ScalarT,FeatureDimension> data_map_;
    cilantro::VectorSet<ScalarT,FeatureDimension> transformed_data_;
    cilantro::ConstVectorSetMatrixMap<ScalarT,FeatureDimension> transformed_data_map_;
};

//class HybridCorrespondenceDistanceEvaluator {
//public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//
//    typedef float InputScalar;
//    typedef float OutputScalar;
//
//    HybridCorrespondenceDistanceEvaluator(const cilantro::PointCloud3f &cloud1,
//                                          const cilantro::PointCloud3f &cloud2)
//            : cloud1(cloud1), cloud2(cloud2)
//    {}
//
//    inline float operator()(size_t i, size_t j, float dist) const {
//        return (std::abs(cloud1.points(2,i) - cloud2.points(2,j)) < 2e-3) ? dist : std::numeric_limits<float>::infinity();
//    }
//
//    const cilantro::PointCloud3f& cloud1;
//    const cilantro::PointCloud3f& cloud2;
//};

//template <class TransformT>
//cilantro::TransformSet<TransformT> getSparseWarpField(const cilantro::PointCloud<typename TransformT::Scalar,TransformT::Dim> &frame1,
//                                                      const cilantro::PointCloud<typename TransformT::Scalar,TransformT::Dim> &frame2,
//                                                      size_t &iter)
//{
//    typedef typename TransformT::Scalar Scalar;
//    enum { Dim = TransformT::Dim };
//
//    Scalar control_res = 0.01;
//    Scalar src_to_control_sigma = 0.02;
//    Scalar regularization_sigma = 0.03;
//
//    Scalar max_correspondence_dist_sq = 0.1*0.1;
//
//    // Get a sparse set of control nodes by downsampling
//    cilantro::VectorSet<Scalar,Dim> control_points = cilantro::PointsGridDownsampler<Scalar,Dim>(frame2.points, control_res).getDownsampledPoints();
//    cilantro::KDTree<Scalar,Dim> control_tree(control_points);
//
//    // Find which control nodes affect each point in src
//    std::vector<cilantro::NeighborSet<Scalar>> src_to_control_nn;
//    control_tree.search(frame2.points, cilantro::KNNNeighborhoodSpecification(3), src_to_control_nn);
//
//    // Get regularization neighborhoods for control nodes
//    std::vector<cilantro::NeighborSet<Scalar>> regularization_nn;
//    control_tree.search(control_points, cilantro::KNNNeighborhoodSpecification(5), regularization_nn);
//
//    cilantro::SimpleCombinedMetricSparseWarpFieldICP<TransformT> icp(frame1.points, frame1.normals, frame2.points, src_to_control_nn, control_points.cols(), regularization_nn);
//
//    icp.correspondenceSearchEngine().setMaxDistance(max_correspondence_dist_sq);
//    icp.controlWeightEvaluator().setSigma(src_to_control_sigma);
//    icp.regularizationWeightEvaluator().setSigma(regularization_sigma);
//
//    icp.setMaxNumberOfIterations(10).setConvergenceTolerance(2.5e-3);
//    icp.setMaxNumberOfGaussNewtonIterations(1).setGaussNewtonConvergenceTolerance(5e-4);
//    icp.setMaxNumberOfConjugateGradientIterations(500).setConjugateGradientConvergenceTolerance(1e-5);
//    icp.setPointToPointMetricWeight(0.1).setPointToPlaneMetricWeight(1.0).setStiffnessRegularizationWeight(200.0);
//    icp.setHuberLossBoundary(1e-2);
//
//    icp.estimate();
//    iter = icp.getNumberOfPerformedIterations();
//
//    return icp.getDenseWarpField();
//}

template <class TransformT>
cilantro::TransformSet<TransformT> getDenseWarpField(const cilantro::PointCloud<typename TransformT::Scalar,TransformT::Dim + 1> &frame1,
                                                     const cilantro::PointCloud<typename TransformT::Scalar,TransformT::Dim + 1> &frame2,
                                                     size_t &iter)
{
    typedef typename TransformT::Scalar Scalar;
    enum { Dim = TransformT::Dim };

    // "Project" to 2D
    cilantro::VectorSet<Scalar,Dim> frame1_points_2d = frame1.points.template topRows<Dim>();
    cilantro::VectorSet<Scalar,Dim> frame2_points_2d = frame2.points.template topRows<Dim>();

    cilantro::VectorSet<Scalar,Dim> frame1_normals_2d = frame1.normals.template topRows<Dim>();
#pragma omp parallel for
    for (size_t i = 0; i < frame1_normals_2d.cols(); i++) {
        frame1_normals_2d.col(i).normalize();
        if (!frame1_normals_2d.col(i).allFinite()) frame1_normals_2d.col(i).setZero();
    }

    Scalar max_correspondence_dist_sq = 0.1*0.1;
    Scalar regularization_sigma = 0.05;

    std::vector<cilantro::NeighborSet<Scalar>> regularization_nn;
    cilantro::KDTree<Scalar,Dim + 1>(frame2.points).search(frame2.points, cilantro::KNNNeighborhoodSpecification(12), regularization_nn);

    HybridPointsAdaptor<float,3> ff1(frame1.points);
    HybridPointsAdaptor<float,3> ff2(frame2.points);

//    cilantro::PointFeaturesAdaptor2f ff1(frame1_points_2d);
//    cilantro::PointFeaturesAdaptor2f ff2(frame2_points_2d);

    cilantro::DistanceEvaluator<float> dist_eval;
    cilantro::CorrespondenceSearchKDTree<decltype(ff1)> corr_engine(ff1, ff2, dist_eval);
//    HybridCorrespondenceDistanceEvaluator dist_eval(frame1, frame2);
//    cilantro::CorrespondenceSearchKDTree<decltype(ff1),cilantro::KDTreeDistanceAdaptors::L2,decltype(ff1), decltype(dist_eval)> corr_engine(ff1, ff2, dist_eval);


    cilantro::UnityWeightEvaluator<float> corr_weight_eval;
    cilantro::RBFKernelWeightEvaluator<float> reg_eval(regularization_sigma);

    cilantro::CombinedMetricDenseWarpFieldICP<TransformT,decltype(corr_engine)> icp(frame1_points_2d, frame1_normals_2d, frame2_points_2d, corr_engine, corr_weight_eval, corr_weight_eval, regularization_nn, reg_eval);

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
    const double frame_duration = 30e-3;
    const double t_skip = -0e-3;

    cilantro::VectorSet3f events;
    readAllEvents3D(argv[1], events);

    double start = 0e-3;

    double start1 = start;
    double end1 = start1 + frame_duration;
    double start2 = end1 + t_skip;
    double end2 = start2 + frame_duration;

    auto ts = events.row(2);

    cilantro::PointCloud3f frame1 = getFrame<3>(events, getIndexByTS(ts, start1), getIndexByTS(ts, end1));
    frame1.points.row(2) = frame1.points.row(2).array() - start1;
    cilantro::PointCloud3f frame2 = getFrame<3>(events, getIndexByTS(ts, start2), getIndexByTS(ts, end2));
    frame2.points.row(2).array() = frame2.points.row(2).array() - start2;

    frame1.estimateNormalsKNN(30);
//    frame2.estimateNormalsKNN(30);


    size_t iter;
    cilantro::Timer timer;
    timer.start();
//    auto wf = getDenseWarpField<cilantro::AffineTransform2f>(frame1, frame2, iter);
    auto wf = getDenseWarpField<cilantro::RigidTransform2f>(frame1, frame2, iter);
//    auto wf = getSparseWarpField<cilantro::AffineTransform2f>(frame1, frame2, iter);
//    auto wf = getSparseWarpField<cilantro::RigidTransform2f>(frame1, frame2, iter);
    timer.stop();
    std::cout << "Total registration time: " << timer.getElapsedTime() << "ms (" << iter << " iterations)" << std::endl;


    cilantro::PointCloud3f frame2_t;
    frame2_t.points.resize(3, wf.size());
    transformPointsHybrid(wf, frame2.points, frame2_t.points);


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
