#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>
#include <apriltag/apriltag_pose.h>

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

#include <iostream>

int main() {
    rs2::pipeline p;
    rs2::config cfg;

    // For right now, turn only one of the infrared cameras
    // cfg.enable_stream(RS2_STREAM_GYRO);
    // cfg.enable_stream(RS2_STREAM_ACCEL);
    // cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
    // cfg.enable_stream(RS2_DEPTH_SENSOR, RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    std::cout << "Enabling infrared streams..." << std::endl;
    // 0 is left, 1 is right, error if both enabled
    //cfg.enable_stream(RS2_STREAM_INFRARED, 0, 1280, 800, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 1280, 800, RS2_FORMAT_Y8, 30);

    auto profile = p.start(cfg);
    auto dev = profile.get_device();

    // Turn off the projector
    std::cout << "Disabling projector..." << std::endl;
    auto depth_sensor = dev.first<rs2::depth_sensor>();
    depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0);    

    // Get the camera intrinsics
    auto right_stream = profile.get_stream(RS2_STREAM_INFRARED, 1)
                                .as<rs2::video_stream_profile>();
    rs2_intrinsics intrinsics = right_stream.get_intrinsics();
    std::cout << "Right IR Camera Intrinsics: " << std::endl;
    std::cout << "  Width: " << intrinsics.width << std::endl;
    std::cout << "  Height: " << intrinsics.height << std::endl;
    std::cout << "  PPX: " << intrinsics.ppx << std::endl;
    std::cout << "  PPY: " << intrinsics.ppy << std::endl;
    std::cout << "  FX: " << intrinsics.fx << std::endl;
    std::cout << "  FY: " << intrinsics.fy << std::endl;
    std::cout << "  Distortion Model: " << intrinsics.model << std::endl;
    std::cout << "  Distortion Coefficients: ";
    for (int i = 0; i < 5; i++) {
        std::cout << intrinsics.coeffs[i] << " ";
    }
    std::cout << std::endl;


    // Prepare undistortion maps
    cv::Mat right_K = (cv::Mat1d(3, 3) << intrinsics.fx, 0, intrinsics.ppx,
                                              0, intrinsics.fy, intrinsics.ppy,
                                              0, 0, 1);
    cv::Mat right_D = (cv::Mat1d(1, 5) << intrinsics.coeffs[0],
                                               intrinsics.coeffs[1],
                                               intrinsics.coeffs[2],
                                               intrinsics.coeffs[3],
                                               intrinsics.coeffs[4]);

    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(right_K, right_D, cv::Mat(),
                                right_K, cv::Size(intrinsics.width, intrinsics.height),
                                CV_32FC1, map1, map2);

    // initialize apriltag stuff
    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);

    td->quad_decimate = 1.0;
    td->quad_sigma = 0.0;
    td->nthreads = 1;
    td->refine_edges = 1;
    
    while (true) {
        rs2::frameset frames = p.wait_for_frames();
        rs2::video_frame right_ir = frames.get_infrared_frame(1);
        if (!right_ir) continue;

        // Calibrate the image
        cv::Mat right_undistorted;
        cv::Mat right_ir_mat(cv::Size(right_ir.get_width(), right_ir.get_height()), CV_8UC1, (void*)right_ir.get_data(), cv::Mat::AUTO_STEP);
        cv::remap(right_ir_mat, right_undistorted, map1, map2, cv::INTER_LINEAR);

        // Detect AprilTags
        image_u8_t im = { .width = right_undistorted.cols,
                          .height = right_undistorted.rows,
                          .stride = right_undistorted.cols,
                          .buf = right_undistorted.data
                        };
        zarray_t *detections = apriltag_detector_detect(td, &im);

        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);

            // Draw detection on the image
            cv::line(right_undistorted, cv::Point(det->p[0][0], det->p[0][1]),
                                         cv::Point(det->p[1][0], det->p[1][1]),
                                         cv::Scalar(255, 0, 0), 2);
            cv::line(right_undistorted, cv::Point(det->p[1][0], det->p[1][1]),
                                         cv::Point(det->p[2][0], det->p[2][1]),
                                         cv::Scalar(255, 0, 0), 2);
            cv::line(right_undistorted, cv::Point(det->p[2][0], det->p[2][1]),
                                         cv::Point(det->p[3][0], det->p[3][1]),
                                         cv::Scalar(255, 0, 0), 2);
            cv::line(right_undistorted, cv::Point(det->p[3][0], det->p[3][1]),
                                         cv::Point(det->p[0][0], det->p[0][1]),
                                         cv::Scalar(255, 0, 0), 2);

            // Draw tag ID
            cv::putText(right_undistorted, std::to_string(det->id),
                        cv::Point(det->c[0], det->c[1]),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0, 255, 0), 2);
        }
        apriltag_detections_destroy(detections);

        cv::imshow("Right IR", right_undistorted);

        if (cv::waitKey(1) == 27) break; // Exit on ESC key
    }

    return 0;
}