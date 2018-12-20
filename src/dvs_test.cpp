#include <cilantro/cilantro.hpp>

int main(int argc, char ** argv) {

    Eigen::MatrixXd data;
    cilantro::readEigenMatrixFromFile(argv[1], data, false);

    cilantro::VectorSet3f events(3, data.rows());
    float mu_x = data.col(1).mean();
    float mu_y = data.col(2).mean();

    events.row(0) = (data.col(1).cast<float>().array() - mu_x)/200.0f;
    events.row(1) = (data.col(2).cast<float>().array() - mu_y)/200.0f;
    events.row(2) = ((data.col(0).array() - data.col(0).minCoeff()).cast<float>()).array() + 1.5f;

    cilantro::KDTree3f tree(events);
    cilantro::NeighborSet<float> nn;
    cilantro::RadiusNeighborhoodSpecification<float> nh(0.0005f);
    cilantro::VectorSet3f filtered(3, events.cols());
    size_t k = 0;
    for (size_t i = 0; i < events.cols(); i++) {
        tree.search(events.col(i), nh, nn);
        if (nn.size() > 30) {
            filtered.col(k++) = events.col(i);
        }
    }
    filtered.conservativeResize(3, k);

    std::cout << "Before: " << events.cols() << ", after: " << filtered.cols() << std::endl;

    pangolin::CreateWindowAndBind("test",1280,480);
    pangolin::Display("multi").SetBounds(0.0, 1.0, 0.0, 1.0).SetLayout(pangolin::LayoutEqual).AddDisplay(pangolin::Display("disp1")).AddDisplay(pangolin::Display("disp2"));

    cilantro::Visualizer viz1("test", "disp1");
    cilantro::Visualizer viz2("test", "disp2");

    cilantro::VectorSet<float,1> event_time(events.row(2));
    cilantro::VectorSet<float,1> filtered_time(filtered.row(2));

    viz1.addObject<cilantro::PointCloudRenderable>("events", events)->setPointValues(event_time);
    viz2.addObject<cilantro::PointCloudRenderable>("filtered", filtered)->setPointValues(filtered_time);

    while (!viz1.wasStopped() && !viz2.wasStopped()) {
        viz1.clearRenderArea();
        viz1.render();
        viz2.render();
        pangolin::FinishFrame();
    }

    return 0;
}
