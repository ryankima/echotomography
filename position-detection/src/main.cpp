#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>
#include <apriltag/apriltag_pose.h>

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <ctime>

struct TagPose3D {
    Eigen::Vector3f center_cm;
    Eigen::Matrix3f rotation;
};

// Board configuration: 10 AprilTags arranged as:
//    9    
// 0 1 2
// 3 4 5
// 6 7 8

// Define 3D positions of AprilTags on the fixed board
// Origin at tag 4 (center), with units in cm
// X-axis pointing right, Y-axis pointing up, Z-axis pointing towards camera
struct BoardConfiguration {
    static constexpr float PI = 3.14159265358979323846f;
    // Tag spacing parameters
    static constexpr float TAG_SIZE_CM = 15.2f;          // Physical tag size
    static constexpr float EDGE_TO_EDGE_SPACING_CM = 9.0f; // Space between tag edges (standard)
    static constexpr float EDGE_TO_EDGE_SPACING_9_TO_1_CM = 10.75f; // Space between tag 9 and 1
    static constexpr float CENTER_TO_CENTER_SPACING_CM = TAG_SIZE_CM + EDGE_TO_EDGE_SPACING_CM;
    static constexpr float CENTER_TO_CENTER_SPACING_9_TO_1_CM = TAG_SIZE_CM + EDGE_TO_EDGE_SPACING_9_TO_1_CM;
    static constexpr float BEND_ANGLE_DEG = 25.0f;       // Bending angle for columns
    static constexpr float BEND_ANGLE_RAD = BEND_ANGLE_DEG * PI / 180.0f;
    
    // Calculate Z offset for bent columns (backward bending)
    static constexpr float BEND_SIN = 0.4226182617f;
    static constexpr float Z_OFFSET_BENT = CENTER_TO_CENTER_SPACING_CM * BEND_SIN;

    static Eigen::Vector3f cmToMeters(const Eigen::Vector3f& value_cm) {
        return value_cm / 100.0f;
    }

    static Eigen::Matrix3f rotationX(float radians) {
        return Eigen::AngleAxisf(radians, Eigen::Vector3f::UnitX()).toRotationMatrix();
    }

    static Eigen::Matrix3f rotationY(float radians) {
        return Eigen::AngleAxisf(radians, Eigen::Vector3f::UnitY()).toRotationMatrix();
    }
    
    // Tag poses (center in cm relative to tag 4 at origin, plus tag orientation)
    static std::map<int, TagPose3D> getTagPoses() {
        std::map<int, TagPose3D> poses;
        
        // Center column (straight): tags 1, 4, 7
        poses[1] = {cmToMeters({0.0f, CENTER_TO_CENTER_SPACING_CM, 0.0f}), rotationX(0.0f)};
        poses[4] = {cmToMeters({0.0f, 0.0f, 0.0f}), rotationX(0.0f)};
        poses[7] = {cmToMeters({0.0f, -CENTER_TO_CENTER_SPACING_CM, 0.0f}), rotationX(0.0f)};
        
        // Left column (bent back): tags 0, 3, 6
        float x_left = -CENTER_TO_CENTER_SPACING_CM;
        float z_left = -Z_OFFSET_BENT;
        poses[0] = {cmToMeters({x_left, CENTER_TO_CENTER_SPACING_CM, z_left}), rotationY(-BEND_ANGLE_RAD)};
        poses[3] = {cmToMeters({x_left, 0.0f, z_left}), rotationY(-BEND_ANGLE_RAD)};
        poses[6] = {cmToMeters({x_left, -CENTER_TO_CENTER_SPACING_CM, z_left}), rotationY(-BEND_ANGLE_RAD)};
        
        // Right column (bent back): tags 2, 5, 8
        float x_right = CENTER_TO_CENTER_SPACING_CM;
        float z_right = -Z_OFFSET_BENT;
        poses[2] = {cmToMeters({x_right, CENTER_TO_CENTER_SPACING_CM, z_right}), rotationY(BEND_ANGLE_RAD)};
        poses[5] = {cmToMeters({x_right, 0.0f, z_right}), rotationY(BEND_ANGLE_RAD)};
        poses[8] = {cmToMeters({x_right, -CENTER_TO_CENTER_SPACING_CM, z_right}), rotationY(BEND_ANGLE_RAD)};
        
        // Top tag 9 (bent back): centered horizontally, above the top row
        // Special spacing: 10.75 cm edge-to-edge from tag 1
        float y_top = CENTER_TO_CENTER_SPACING_9_TO_1_CM + CENTER_TO_CENTER_SPACING_CM;
        float z_top = -Z_OFFSET_BENT;
        poses[9] = {cmToMeters({0.0f, y_top, z_top}), rotationX(-BEND_ANGLE_RAD)};
        
        return poses;
    }

    static std::array<Eigen::Vector3f, 4> tagCornerOffsetsMeters() {
        const float half_tag_m = TAG_SIZE_CM * 0.5f / 100.0f;
        return {
            Eigen::Vector3f(-half_tag_m, -half_tag_m, 0.0f),
            Eigen::Vector3f( half_tag_m, -half_tag_m, 0.0f),
            Eigen::Vector3f( half_tag_m,  half_tag_m, 0.0f),
            Eigen::Vector3f(-half_tag_m,  half_tag_m, 0.0f)
        };
    }
};

struct PoseFilter {
    bool initialized = false;
    double last_gyro_timestamp_ms = 0.0;
    Eigen::Vector3f position_m = Eigen::Vector3f::Zero();
    Eigen::Quaternionf orientation = Eigen::Quaternionf::Identity();
    Eigen::Vector3f latest_gyro_rad_s = Eigen::Vector3f::Zero();
    Eigen::Vector3f latest_accel_m_s2 = Eigen::Vector3f::Zero();
    bool has_gyro = false;
    bool has_accel = false;

    static Eigen::Quaternionf gyroDelta(const Eigen::Vector3f& gyro_rad_s, float dt_seconds) {
        float angle = gyro_rad_s.norm() * dt_seconds;
        if (angle <= 1e-7f) {
            return Eigen::Quaternionf::Identity();
        }

        Eigen::Vector3f axis = gyro_rad_s.normalized();
        return Eigen::Quaternionf(Eigen::AngleAxisf(angle, axis));
    }

    void integrateGyro(double timestamp_ms) {
        if (!has_gyro) {
            return;
        }

        if (last_gyro_timestamp_ms > 0.0 && timestamp_ms > last_gyro_timestamp_ms) {
            float dt_seconds = static_cast<float>((timestamp_ms - last_gyro_timestamp_ms) / 1000.0);
            if (dt_seconds > 0.0f && dt_seconds < 0.1f) {
                orientation = (orientation * gyroDelta(latest_gyro_rad_s, dt_seconds)).normalized();
            }
        }

        last_gyro_timestamp_ms = timestamp_ms;
    }

    void fuseVision(const Eigen::Vector3f& vision_position_m, const Eigen::Quaternionf& vision_orientation) {
        if (!initialized) {
            position_m = vision_position_m;
            orientation = vision_orientation.normalized();
            initialized = true;
            return;
        }

        float gyro_rate = has_gyro ? latest_gyro_rad_s.norm() : 0.0f;
        float accel_deviation = has_accel ? std::fabs(latest_accel_m_s2.norm() - 9.80665f) : 0.0f;

        float position_alpha = std::clamp(0.12f + 0.015f * gyro_rate + 0.01f * accel_deviation, 0.06f, 0.35f);
        float orientation_alpha = std::clamp(0.18f + 0.02f * gyro_rate, 0.08f, 0.55f);

        position_m = (1.0f - position_alpha) * position_m + position_alpha * vision_position_m;
        orientation = orientation.slerp(orientation_alpha, vision_orientation).normalized();
    }
};

// Structure to hold synchronized frame data
struct SyncFrameData {
    cv::Mat camera_frame;
    cv::Mat capture_frame;
    uint64_t timestamp;
    int frame_count;
};

// Thread-safe queue for synchronized frames
class SyncFrameQueue {
private:
    std::queue<SyncFrameData> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    const size_t max_size_ = 30; // Maximum queue size
    
public:
    void push(const SyncFrameData& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_) {
            queue_.pop(); // Drop oldest frame if queue is full
        }
        queue_.push(data);
        cv_.notify_one();
    }
    
    bool try_pop(SyncFrameData& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        data = queue_.front();
        queue_.pop();
        return true;
    }
    
    size_t size() {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

// Global state for synchronized recording
struct RecordingState {
    bool is_recording = false;
    std::mutex mutex;
    std::condition_variable cv;
    int camera_frame_count = 0;
    int capture_frame_count = 0;
};

int main() {
    // ==================== RealSense Pipeline Setup ====================
    rs2::pipeline p;
    rs2::config cfg;

    std::cout << "Enabling video stream..." << std::endl;
    cfg.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_RGB8, 30);
    cfg.enable_stream(RS2_STREAM_GYRO);
    cfg.enable_stream(RS2_STREAM_ACCEL);
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
    std::cout << "Camera Intrinsics: " << std::endl;
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
    cv::Mat camera_K = (cv::Mat1d(3, 3) << intrinsics.fx, 0, intrinsics.ppx,
                                              0, intrinsics.fy, intrinsics.ppy,
                                              0, 0, 1);
    cv::Mat camera_D = (cv::Mat1d(1, 5) << intrinsics.coeffs[0],
                                               intrinsics.coeffs[1],
                                               intrinsics.coeffs[2],
                                               intrinsics.coeffs[3],
                                               intrinsics.coeffs[4]);

    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(camera_K, camera_D, cv::Mat(),
                                camera_K, cv::Size(intrinsics.width, intrinsics.height),
                                CV_32FC1, map1, map2);

    // ==================== USB Capture Card Setup ====================
    std::cout << "\nAttempting to open USB capture card..." << std::endl;
    cv::VideoCapture capture_card;
    int capture_width = 1920, capture_height = 1080;
    
    // Try to find the capture card device
    // On Windows with DirectShow, typically index 1 or higher if index 0 is the webcam
    // You may need to adjust the device index based on your system
    for (int i = 1; i <= 3; i++) {
        capture_card.open(i, cv::CAP_DSHOW);
        if (capture_card.isOpened()) {
            std::cout << "Found capture card at device index " << i << std::endl;
            capture_card.set(cv::CAP_PROP_FRAME_WIDTH, capture_width);
            capture_card.set(cv::CAP_PROP_FRAME_HEIGHT, capture_height);
            capture_card.set(cv::CAP_PROP_FPS, 30);
            break;
        }
    }
    
    if (!capture_card.isOpened()) {
        std::cout << "Warning: Could not open capture card. Recording camera only." << std::endl;
    }

    // ==================== AprilTag Setup ====================
    std::cout << "\nInitializing AprilTag detector..." << std::endl;
    apriltag_family_t *tf = tag36h11_create();
    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);

    td->quad_decimate = 1.0;
    td->quad_sigma = 0.0;
    td->nthreads = 4;
    td->refine_edges = 1;

    // Get board tag positions
    auto board_tag_poses = BoardConfiguration::getTagPoses();
    auto tag_corner_offsets_m = BoardConfiguration::tagCornerOffsetsMeters();

    // ==================== Video Recording Setup ====================
    std::cout << "\nSetting up video recording..." << std::endl;
    cv::VideoWriter camera_writer, capture_writer;
    RecordingState recording_state;
    SyncFrameQueue frame_queue;
    
    std::string output_dir = "./recordings/";
    system("mkdir recordings 2>nul"); // Create directory if it doesn't exist (Windows)
    
    bool recording_enabled = true;
    int frame_count = 0;

    // ==================== Main Processing Loop ====================
    std::cout << "\nStarting main loop. Press:" << std::endl;
    std::cout << "  'r' to start/stop recording" << std::endl;
    std::cout << "  'ESC' to exit" << std::endl;

    cv::Mat gray_mat, undistorted, color_mat, capture_frame;
    Eigen::Vector3f camera_position = Eigen::Vector3f::Zero();
    Eigen::Quaternionf camera_orientation = Eigen::Quaternionf::Identity();
    PoseFilter pose_filter;
    
    while (true) {
        // Get frame from RealSense camera
        rs2::frameset frames = p.wait_for_frames();
        rs2::video_frame color = frames.get_color_frame();
        if (!color) continue;

        rs2::frame gyro_frame = frames.first_or_default(RS2_STREAM_GYRO);
        if (gyro_frame) {
            auto motion = gyro_frame.as<rs2::motion_frame>().get_motion_data();
            pose_filter.latest_gyro_rad_s = Eigen::Vector3f(motion.x, motion.y, motion.z);
            pose_filter.has_gyro = true;
            pose_filter.integrateGyro(gyro_frame.get_timestamp());
        }

        rs2::frame accel_frame = frames.first_or_default(RS2_STREAM_ACCEL);
        if (accel_frame) {
            auto motion = accel_frame.as<rs2::motion_frame>().get_motion_data();
            pose_filter.latest_accel_m_s2 = Eigen::Vector3f(motion.x, motion.y, motion.z);
            pose_filter.has_accel = true;
        }
    
        color_mat = cv::Mat(cv::Size(color.get_width(), color.get_height()), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
        cv::cvtColor(color_mat, gray_mat, cv::COLOR_RGB2GRAY);
        cv::remap(gray_mat, undistorted, map1, map2, cv::INTER_LINEAR);
        
        // Get frame from capture card (non-blocking)
        bool capture_available = false;
        if (capture_card.isOpened()) {
            capture_available = capture_card.read(capture_frame);
        }
        
        // ==================== AprilTag Detection ====================
        image_u8_t im = {
            .width = undistorted.cols,
            .height = undistorted.rows,
            .stride = (int)undistorted.step,
            .buf = undistorted.data
        };
    
        zarray_t *detections = apriltag_detector_detect(td, &im);
        
        // ==================== Camera Pose Estimation ====================
        // Collect detected tag information
        std::vector<cv::Point2f> image_points;
        std::vector<cv::Point3f> object_points;
        std::map<int, apriltag_detection_t*> detected_map;
        
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);
            detected_map[det->id] = det;
            
            // Check if this tag is part of our board
            auto pose_it = board_tag_poses.find(det->id);
            if (pose_it != board_tag_poses.end()) {
                const TagPose3D& tag_pose = pose_it->second;
                for (size_t corner_idx = 0; corner_idx < tag_corner_offsets_m.size(); ++corner_idx) {
                    Eigen::Vector3f corner_m = tag_pose.center_cm + tag_pose.rotation * tag_corner_offsets_m[corner_idx];
                    object_points.push_back(cv::Point3f(corner_m.x(), corner_m.y(), corner_m.z()));
                    image_points.push_back(cv::Point2f(det->p[corner_idx][0], det->p[corner_idx][1]));
                }
            }
        }
        
        // Estimate camera pose if we have enough detections (at least 4 tags)
        if (object_points.size() >= 8) {
            cv::Mat rvec, tvec;
            cv::Mat inliers;
            
            // Use RANSAC for robust pose estimation
            bool success = cv::solvePnPRansac(object_points, image_points, camera_K, camera_D,
                                             rvec, tvec, false, 100, 4.0f, 0.99, inliers);
            
            if (success && inliers.rows >= 8) {
                // Convert rotation vector to rotation matrix
                cv::Mat R;
                cv::Rodrigues(rvec, R);

                // solvePnP gives the board pose in the camera frame:
                // X_cam = R * X_board + t
                // Camera position in board coordinates is therefore -R^T * t.
                Eigen::Matrix3f R_eigen;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        R_eigen(i, j) = static_cast<float>(R.at<double>(i, j));
                    }
                }

                Eigen::Vector3f t_eigen(
                    static_cast<float>(tvec.at<double>(0)),
                    static_cast<float>(tvec.at<double>(1)),
                    static_cast<float>(tvec.at<double>(2)));

                Eigen::Vector3f vision_position = -(R_eigen.transpose() * t_eigen);

                // Extract camera orientation and smooth with IMU-guided filtering
                Eigen::Quaternionf vision_orientation(R_eigen);
                pose_filter.fuseVision(vision_position, vision_orientation);
                camera_position = pose_filter.position_m;
                camera_orientation = pose_filter.orientation;
                
                // Debug: Print camera pose
                if (frame_count % 30 == 0) {
                    std::cout << "Camera Position: (" << camera_position.x() << ", " 
                              << camera_position.y() << ", " << camera_position.z() << ") m" << std::endl;
                    if (pose_filter.has_gyro) {
                        std::cout << "  Gyro norm: " << pose_filter.latest_gyro_rad_s.norm() << " rad/s" << std::endl;
                    }
                    if (pose_filter.has_accel) {
                        std::cout << "  Accel norm: " << pose_filter.latest_accel_m_s2.norm() << " m/s^2" << std::endl;
                    }
                }
            }
        }
        
        // ==================== Visualization ====================
        cv::Mat display_frame = undistorted.clone();
        cv::cvtColor(display_frame, display_frame, cv::COLOR_GRAY2BGR);
        
        cv::Scalar blue(255, 0, 0), green(0, 255, 0), red(0, 0, 255), yellow(0, 255, 255);
        
        // Draw detected tags
        for (const auto& [tag_id, det] : detected_map) {
            // Draw tag corners
            cv::Scalar color = (board_tag_poses.find(tag_id) != board_tag_poses.end()) ? green : blue;
            for (int j = 0; j < 4; j++) {
                int k = (j+1)%4;
                cv::line(display_frame,
                         cv::Point(det->p[j][0], det->p[j][1]),
                         cv::Point(det->p[k][0], det->p[k][1]),
                         color, 2);
            }
            
            // Draw tag ID
            cv::putText(display_frame, std::to_string(tag_id),
                        cv::Point(det->c[0], det->c[1]),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
        }
        
        // Display camera pose on frame
        std::string pose_text = "Pos: (" + std::to_string((int)(camera_position.x()*1000)) + ", " +
                       std::to_string((int)(camera_position.y()*1000)) + ", " +
                       std::to_string((int)(camera_position.z()*1000)) + ") mm";
        cv::putText(display_frame, pose_text, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, red, 2);

        std::string imu_text = "IMU |g|=" + std::to_string(pose_filter.has_gyro ? pose_filter.latest_gyro_rad_s.norm() : 0.0f) +
                       " rad/s  |a|=" + std::to_string(pose_filter.has_accel ? pose_filter.latest_accel_m_s2.norm() : 0.0f) +
                       " m/s^2";
        cv::putText(display_frame, imu_text, cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, yellow, 2);
        
        std::string tag_count = "Tags detected: " + std::to_string(zarray_size(detections));
        cv::putText(display_frame, tag_count, cv::Point(10, 90),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, red, 2);
        
        std::string rec_status = recording_state.is_recording ? "REC (camera,capture)" : "IDLE";
        cv::Scalar rec_color = recording_state.is_recording ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::putText(display_frame, rec_status, cv::Point(10, 120),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2);
        
        // ==================== Video Recording ====================
        {
            std::unique_lock<std::mutex> lock(recording_state.mutex);
            
            if (recording_state.is_recording) {
                // Initialize writers on first recording frame
                if (!camera_writer.isOpened()) {
                    std::string timestamp = std::to_string(std::time(nullptr));
                    std::string camera_output = output_dir + "camera_" + timestamp + ".mp4";
                    std::string capture_output = output_dir + "capture_" + timestamp + ".mp4";
                    
                    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                    camera_writer.open(camera_output, fourcc, 30, cv::Size(display_frame.cols, display_frame.rows));
                    
                    if (capture_available) {
                        capture_writer.open(capture_output, fourcc, 30, cv::Size(capture_frame.cols, capture_frame.rows));
                        std::cout << "Recording to " << capture_output << std::endl;
                    }
                    
                    std::cout << "Recording to " << camera_output << std::endl;
                }
                
                // Write frames
                if (camera_writer.isOpened()) {
                    camera_writer.write(display_frame);
                    recording_state.camera_frame_count++;
                }
                
                if (capture_writer.isOpened() && capture_available) {
                    capture_writer.write(capture_frame);
                    recording_state.capture_frame_count++;
                }
            } else {
                // Close writers when not recording
                if (camera_writer.isOpened() || capture_writer.isOpened()) {
                    if (camera_writer.isOpened()) {
                        camera_writer.release();
                        std::cout << "Closed camera recording (" << recording_state.camera_frame_count << " frames)" << std::endl;
                    }
                    if (capture_writer.isOpened()) {
                        capture_writer.release();
                        std::cout << "Closed capture recording (" << recording_state.capture_frame_count << " frames)" << std::endl;
                    }
                    recording_state.camera_frame_count = 0;
                    recording_state.capture_frame_count = 0;
                }
            }
        }
        
        apriltag_detections_destroy(detections);
        
        // ==================== Display ====================
        cv::imshow("Camera Feed with Pose", display_frame);
        if (capture_available) {
            cv::imshow("Capture Card Feed", capture_frame);
        }
        
        // ==================== Keyboard Input ====================
        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            break;
        } else if (key == 'r' || key == 'R') {
            {
                std::unique_lock<std::mutex> lock(recording_state.mutex);
                recording_state.is_recording = !recording_state.is_recording;
                if (recording_state.is_recording) {
                    std::cout << "\nStarting synchronized recording..." << std::endl;
                } else {
                    std::cout << "\nStopping synchronized recording..." << std::endl;
                }
            }
        }
        
        frame_count++;
    }
    
    // ==================== Cleanup ====================
    {
        std::unique_lock<std::mutex> lock(recording_state.mutex);
        recording_state.is_recording = false;
    }
    
    if (camera_writer.isOpened()) camera_writer.release();
    if (capture_writer.isOpened()) capture_writer.release();
    if (capture_card.isOpened()) capture_card.release();
    
    p.stop();
    apriltag_detector_destroy(td);
    tag36h11_destroy(tf);
    
    std::cout << "\nShutdown complete." << std::endl;
    return 0;
}