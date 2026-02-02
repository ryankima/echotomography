#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>
#include <apriltag/apriltag_pose.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>

#include <librealsense2/rs.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>

const double tag_size = 0.0508; // meters

int main() {
    rs2::pipeline p;
    rs2::config cfg;

    std::cout << "Enabling video stream..." << std::endl;
    // For right now, turn only one of the infrared cameras
    // cfg.enable_stream(RS2_STREAM_GYRO);
    // cfg.enable_stream(RS2_STREAM_ACCEL);
    cfg.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_RGB8, 30);
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
    td->nthreads = 4;
    td->refine_edges = 1;

    cv::Mat gray_mat, undistorted, color_mat;

    // store previous pose for temporal stability
    cv::Vec3d prev_rvec(0, 0, 0), prev_tvec(0, 0, 0.3);
    bool has_prev_pose = false;

    // position and orientation mapping of tags, 6 tags on the cube with 50.8 mm sides
    // Cube net layout:    2
    //                  0 1 4 5
    //                     3
    std::vector<std::pair<cv::Vec3d, cv::Quatd>> tag_map = {
        {cv::Vec3d(-0.0254, 0.0, 0.0), cv::Quatd(0.707, 0.0, 0.707, 0.0)},    // Tag 0: left face
        {cv::Vec3d(0.0, 0.0, 0.0254), cv::Quatd(1.0, 0.0, 0.0, 0.0)},         // Tag 1: front face
        {cv::Vec3d(0.0, 0.0254, 0.0), cv::Quatd(0.707, -0.707, 0.0, 0.0)},    // Tag 2: top face
        {cv::Vec3d(0.0, -0.0254, 0.0), cv::Quatd(0.707, 0.707, 0.0, 0.0)},    // Tag 3: bottom face
        {cv::Vec3d(0.0254, 0.0, 0.0), cv::Quatd(0.707, 0.0, -0.707, 0.0)},    // Tag 4: right face
        {cv::Vec3d(0.0, 0.0, -0.0254), cv::Quatd(0.0, 0.0, 1.0, 0.0)},        // Tag 5: back face
    };

    std::vector<cv::Point3d> tag_corners = {
        cv::Point3d(-tag_size/2, -tag_size/2, 0),
        cv::Point3d( tag_size/2, -tag_size/2, 0),
        cv::Point3d( tag_size/2,  tag_size/2, 0),
        cv::Point3d(-tag_size/2,  tag_size/2, 0)
    };

    // open output file for pose data
    std::ofstream pose_file("probe_poses.csv");
    pose_file << "timestamp,frame_number,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,"
              << "tags_visible,total_points,reprojection_error" << std::endl;
    
    int frame_number = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (true) {
        rs2::frameset frames = p.wait_for_frames();
        rs2::video_frame color = frames.get_color_frame();
        if (!color) continue;

        frame_number++;
        auto current_time = std::chrono::high_resolution_clock::now();
        double timestamp = std::chrono::duration<double>(current_time - start_time).count();
    
        color_mat = cv::Mat(cv::Size(color.get_width(), color.get_height()), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
        
        cv::cvtColor(color_mat, gray_mat, cv::COLOR_RGB2GRAY);
    
        cv::remap(gray_mat, undistorted, map1, map2, cv::INTER_LINEAR);
    
        image_u8_t im = {
            .width = undistorted.cols,
            .height = undistorted.rows,
            .stride = (int)undistorted.step,
            .buf = undistorted.data
        };
    
        zarray_t *detections = apriltag_detector_detect(td, &im);
    
        std::vector<cv::Point3d> object_points;
        std::vector<cv::Point2d> image_points;
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);

            // skip if tag ID is out of bounds
            if (det->id < 0 || det->id >= tag_map.size()) continue;

            auto tag_transform = tag_map[det->id];
            cv::Vec3d tag_t = tag_transform.first;
            cv::Quatd tag_q = tag_transform.second;
            cv::Matx33d R_tag = tag_q.toRotMat3x3();
    
            for (int j = 0; j < 4; j++) {
                cv::Vec3d local_corner = cv::Vec3d(tag_corners[j].x, tag_corners[j].y, tag_corners[j].z);
                cv::Vec3d object_corner = R_tag * local_corner + tag_t;
                object_points.push_back(cv::Point3d(object_corner[0], object_corner[1], object_corner[2]));
                image_points.push_back(cv::Point2d(det->p[j][0], det->p[j][1]));
            }
        }
        apriltag_detections_destroy(detections);

        if (object_points.size() >= 4) {
            cv::Vec3d rvec, tvec;
            cv::Mat zero_dist = cv::Mat::zeros(1, 5, CV_64F);
            
            bool success;
            if (has_prev_pose) {
                // iterative refinement with previous pose as initial guess
                rvec = prev_rvec;
                tvec = prev_tvec;
                success = cv::solvePnP(object_points, image_points, right_K, zero_dist, 
                                      rvec, tvec, true, cv::SOLVEPNP_ITERATIVE);
            } else {
                // first frame or after tracking loss - use RANSAC
                std::vector<int> inliers;
                success = cv::solvePnPRansac(object_points, image_points, right_K, zero_dist, 
                                            rvec, tvec, false, 100, 8.0, 0.99, inliers);
            }
            
            if (success) {
                // update stored pose for next frame
                prev_rvec = rvec;
                prev_tvec = tvec;
                has_prev_pose = true;
                
                // convert rotation vector to quaternion
                cv::Matx33d rotation_matrix;
                cv::Rodrigues(rvec, rotation_matrix);
                cv::Quatd quat = cv::Quatd::createFromRotMat(rotation_matrix);
                
                // calculate reprojection error for quality metric
                std::vector<cv::Point2d> reprojected_points;
                cv::Mat zero_dist = cv::Mat::zeros(1, 5, CV_64F);
                cv::projectPoints(object_points, rvec, tvec, right_K, zero_dist, reprojected_points);
                double reprojection_error = 0.0;
                for (size_t i = 0; i < image_points.size(); i++) {
                    cv::Point2d diff = image_points[i] - reprojected_points[i];
                    reprojection_error += std::sqrt(diff.x * diff.x + diff.y * diff.y);
                }
                reprojection_error /= image_points.size();
                
                // TODO: apply probe offset transform here (placeholder)
                // cv::Vec3d probe_pos = apply_probe_offset(tvec, rotation_matrix);
                
                // write pose data to file
                pose_file << std::fixed << std::setprecision(6) << timestamp << ","
                         << frame_number << ","
                         << tvec[0] << "," << tvec[1] << "," << tvec[2] << ","
                         << quat.w << "," << quat.x << "," << quat.y << "," << quat.z << ","
                         << zarray_size(detections) << "," << object_points.size() << ","
                         << reprojection_error << std::endl;
                
                // display tracking info
                cv::putText(undistorted, 
                    "Tags: " + std::to_string(zarray_size(detections)) + " Points: " + std::to_string(object_points.size()),
                    cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                cv::putText(undistorted, 
                    "Error: " + cv::format("%.2f px", reprojection_error),
                    cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                
                std::vector<cv::Point3d> axis_points = {
                    cv::Point3d(0, 0, 0),        // origin
                    cv::Point3d(0.05, 0, 0),     // X axis (red)
                    cv::Point3d(0, 0.05, 0),     // Y axis (green)
                    cv::Point3d(0, 0, 0.05)      // Z axis (blue)
                };
                
                std::vector<cv::Point2d> image_axis_points;
                cv::projectPoints(axis_points, rvec, tvec, right_K, zero_dist, image_axis_points);
                
                cv::line(undistorted, image_axis_points[0], image_axis_points[1], cv::Scalar(0, 0, 255), 3); // X = Red
                cv::line(undistorted, image_axis_points[0], image_axis_points[2], cv::Scalar(0, 255, 0), 3); // Y = Green
                cv::line(undistorted, image_axis_points[0], image_axis_points[3], cv::Scalar(255, 0, 0), 3); // Z = Blue
                
                cv::putText(undistorted, 
                    "Pos: " + cv::format("%.3f, %.3f, %.3f", tvec[0], tvec[1], tvec[2]),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                cv::putText(undistorted, 
                    "Rot: " + cv::format("%.3f, %.3f, %.3f", rvec[0], rvec[1], rvec[2]),
                    cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            }
        } else {
            // Reset tracking when we lose sight
            has_prev_pose = false;
        }

    
        cv::imshow("Color", undistorted);
    
        if (cv::waitKey(1) == 27) break;
    }
    
    pose_file.close();
    std::cout << "Pose data saved to probe_poses.csv" << std::endl;

    return 0;
}