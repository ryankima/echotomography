#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>
#include <apriltag/apriltag_pose.h>

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

#include <iostream>

int main() {
    rs2::pipeline p;
    rs2::config cfg;

    std::cout << "Enabling video stream..." << std::endl;
    // For right now, turn only one of the infrared cameras
    // cfg.enable_stream(RS2_STREAM_GYRO);
    // cfg.enable_stream(RS2_STREAM_ACCEL);
    cfg.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_BGR8, 30);
    // cfg.enable_stream(RS2_DEPTH_SENSOR, RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    // 0 is left, 1 is right, error if both enabled
    //cfg.enable_stream(RS2_STREAM_INFRARED, 0, 1280, 800, RS2_FORMAT_Y8, 30);
    //cfg.enable_stream(RS2_STREAM_INFRARED, 1, 1280, 800, RS2_FORMAT_Y8, 30);

    auto profile = p.start(cfg);
    auto dev = profile.get_device();

    // Turn off the projector
    std::cout << "Disabling projector..." << std::endl;
    auto depth_sensor = dev.first<rs2::depth_sensor>();
    depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0);    

    // Get the camera intrinsics
    auto stream = profile.get_stream(RS2_STREAM_COLOR)
                                .as<rs2::video_stream_profile>();
    rs2_intrinsics intrinsics = stream.get_intrinsics();
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
        rs2::video_frame color = frames.get_color_frame();
        if (!color) continue;
    
        cv::Mat color_mat(cv::Size(color.get_width(), color.get_height()), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat gray_mat;
        cv::cvtColor(color_mat, gray_mat, cv::COLOR_RGB2GRAY);
    
        cv::Mat undistorted;
        cv::remap(gray_mat, undistorted, map1, map2, cv::INTER_LINEAR);
    
        image_u8_t im = {
            .width = undistorted.cols,
            .height = undistorted.rows,
            .stride = (int)undistorted.step,
            .buf = undistorted.data
        };
    
        zarray_t *detections = apriltag_detector_detect(td, &im);
    
        cv::Scalar blue(255,0,0), green(0,255,0);
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);
    
            for (int j = 0; j < 4; j++) {
                int k = (j+1)%4;
                cv::line(undistorted,
                         cv::Point(det->p[j][0], det->p[j][1]),
                         cv::Point(det->p[k][0], det->p[k][1]),
                         blue, 2);
            }
    
            cv::putText(undistorted, std::to_string(det->id),
                        cv::Point(det->c[0], det->c[1]),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, green, 2);
        }
        apriltag_detections_destroy(detections);
    
        cv::imshow("Color", undistorted);
    
        if (cv::waitKey(1) == 27) break;
    }
    

    return 0;
}