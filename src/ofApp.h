#pragma once

#include "ofMain.h"
#include "Eigen/Core"

class ofApp : public ofBaseApp{
    
public:
    void setup();
    void update();
    void draw();
    
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void mouseEntered(int x, int y);
    void mouseExited(int x, int y);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
    
    int fps;
    int beforeFps;

    Eigen::Matrix<float, 3, 1> input;
    Eigen::Matrix<float, 3, 3> input_hidden;
    Eigen::Matrix<float, 3, 1> hidden;
    Eigen::Matrix<float, 3, 3> hidden_output;
    Eigen::Matrix<float, 3, 1> output;
    
    Eigen::Matrix<float, 3, 1> teacher;
    Eigen::Matrix<float, 3, 1> output_error;
    Eigen::Matrix<float, 3, 1> hidden_error;
    
    Eigen::Matrix<float, 3, 1> one;
};
