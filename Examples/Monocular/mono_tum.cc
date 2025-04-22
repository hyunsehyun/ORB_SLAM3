#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Dense>
#include "System.h"
#include <Eigen/Geometry>

namespace fs = std::filesystem;
using namespace std;

void LoadImagesFromFolder(const string &folder, vector<string> &image_paths, vector<double> &timestamps, double fps);

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_image_folder" << endl;
        return 1;
    }

    // Load image filenames and fake timestamps
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string folder = string(argv[3]);

    // Get FPS from yaml
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings file" << endl;
        return -1;
    }
    double fps = fsSettings["Camera.fps"];
    LoadImagesFromFolder(folder, vstrImageFilenames, vTimestamps, fps);

    int nImages = vstrImageFilenames.size();
    if (nImages == 0)
    {
        cerr << "ERROR: No images found in folder: " << folder << endl;
        return -1;
    }

    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    vector<float> vTimesTrack(nImages);
    ofstream pose_file("AllFrameTrajectory.txt");
    pose_file << fixed << setprecision(6);

    for (int ni = 0; ni < nImages; ni++)
    {
        string img_path = folder + "/" + vstrImageFilenames[ni];
        cv::Mat im = cv::imread(img_path, cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (im.empty())
        {
            cerr << "Failed to load image at: " << img_path << endl;
            continue;
        }

        if (imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

        auto t1 = chrono::steady_clock::now();
        SLAM.TrackMonocular(im, tframe);
        auto t2 = chrono::steady_clock::now();

        // Get pose after tracking
        Sophus::SE3f Tcw = SLAM.GetCurrentCameraPose();

        // identity pose와 비교하여 유효성 검사
        if (!Tcw.matrix().isApprox(Sophus::SE3f().matrix()))
        {
            // SE3의 inverse → Twc
            Sophus::SE3f Twc = Tcw.inverse();

            Eigen::Vector3f t = Twc.translation();            // 3D 위치
            Eigen::Quaternionf q = Twc.unit_quaternion();     // 회전 → 쿼터니언

            pose_file << std::fixed << std::setprecision(6) << tframe << " "
                    << std::setprecision(9)
                    << t.x() << " " << t.y() << " " << t.z() << " "
                    << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
        double ttrack = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        double T = (ni < nImages - 1) ? vTimestamps[ni + 1] - tframe : vTimestamps[ni] - vTimestamps[ni - 1];
        if (ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    pose_file.close();
    SLAM.Shutdown();

    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0.0f);
    cout << "-------" << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    return 0;
}

void LoadImagesFromFolder(const string &folder, vector<string> &image_paths, vector<double> &timestamps, double fps)
{
    image_paths.clear();
    timestamps.clear();

    for (const auto &entry : fs::directory_iterator(folder))
    {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
            image_paths.push_back(entry.path().filename().string());
    }

    sort(image_paths.begin(), image_paths.end());

    double timestamp = 0.0;
    for (size_t i = 0; i < image_paths.size(); ++i)
    {
        timestamps.push_back(timestamp);
        timestamp += 1.0 / fps;
    }
}
