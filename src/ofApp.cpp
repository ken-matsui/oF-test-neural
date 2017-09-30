#include "ofApp.h"

// シグモイド関数
auto sigmoid = [](float x){return (1/(1 + exp(-x)));};


void ofApp::setup(){
    // Setup the screen.
    ofSetVerticalSync(true);
    ofEnableBlendMode(OF_BLENDMODE_ADD);
    ofSetFrameRate(60);
    ofBackground(0);
    
    // 円の解像度
    ofSetCircleResolution(120);
    
    // 行列の初期化
    input << 0.9, 0.1, 0.8;
    input_hidden << ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)), ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)),
    ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)), ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)),
    ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)), ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)),
    ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)), ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)),
    ofRandom(-1/sqrt(3.0), 1/sqrt(3.0));
    hidden_output << ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)), ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)),
    ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)), ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)),
    ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)), ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)),
    ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)), ofRandom(-1/sqrt(3.0), 1/sqrt(3.0)),
    ofRandom(-1/sqrt(3.0), 1/sqrt(3.0));
    // 教師データの設定
    teacher << 0.99, 0.99, 0.01;
    one.setOnes();
}

void ofApp::update(){
    // Setup the fps and time.
    fps = ceil(ofGetFrameRate());
    //    beforeFps = fps;
    float time = ofGetElapsedTimef();
    ofSetWindowTitle("fps : "+ofToString(fps)+"  time : "+ofToString(ceil(time)));
    
    for (int i = 0; i < 10; i++){
        // 隠れ層
        hidden = input_hidden * input;
        for (int i = 0; i < 3; i++){
            hidden.coeffRef(i, 0) = sigmoid(hidden.coeffRef(i, 0));
        }
        // 出力層
        output = hidden_output * hidden;
        for (int i = 0; i < 3; i++){
            output.coeffRef(i, 0) = sigmoid(output.coeffRef(i, 0));
        }
        
        
        // 勾配降下法---------------------------------------------------
        // 出力値の誤差
        output_error = teacher - output;
        // 隠れ層値の誤差
        hidden_error = hidden_output.transpose() * output_error;
        
        // 学習がうまくいくときといかないときがある！初期の重み値による違いでローカルミニマムに嵌まっている．
        // 隠れ層と出力層の結合重み値を更新 学習係数 : 0.5
        hidden_output += 0.5 * ((output_error.transpose() * output * (one - output)) * hidden.transpose());
        // 入力層と隠れ層の結合重み値を更新
        input_hidden += 0.5 * ((hidden_error.transpose() * hidden * (one - hidden)) * input.transpose());
    }
}

void ofApp::draw(){
    ofNoFill();
    ofSetColor(255);
    
    ofPushMatrix();
    ofTranslate(ofGetWidth() / 2, ofGetHeight() / 2);
    
    // ニューラルネットの可視化
    for (int i = 0; i < 3; i++){ // 縦のニューロン数
        for(int j = 0; j < 3; j++){ // 横のニューロン数(層数)
            ofDrawCircle(-230 + j * 230, -180 + i * 180, 30);
        }
    }
    for (int i = 0; i < 3; i++){ // 縦のニューロン数
        for(int j = 0; j < 3-1; j++){ // 横のニューロン数(層数) -1
            ofDrawLine(-200 + j * 200, -180 + i * 180, -200 + (j+1) * 200, -180 + 0 * 180);
            ofDrawLine(-200 + j * 200, -180 + i * 180, -200 + (j+1) * 200, -180 + 1 * 180);
            ofDrawLine(-200 + j * 200, -180 + i * 180, -200 + (j+1) * 200, -180 + 2 * 180);
        }
    }
    
    // データの可視化
    for (int i = 0; i < 3; i++){
        // 入力値
        ofDrawBitmapString(ofToString(input.coeffRef(i, 0)), -300, -180 + i*180);
        
        // input_hidden結合重み値
        ofDrawBitmapString(ofToString(input_hidden.coeffRef(i, 0)), -200, -180 + i*30);
        ofDrawBitmapString(ofToString(input_hidden.coeffRef(i, 1)), -200, 0 + i*30);
        ofDrawBitmapString(ofToString(input_hidden.coeffRef(i, 2)), -200, 180 + i*30);
        
        // 隠れ層値
        ofDrawBitmapString(ofToString(hidden.coeffRef(i, 0)), -70, -180 + i*180);
        
        // hidden_output結合重み値
        ofDrawBitmapString(ofToString(hidden_output.coeffRef(i, 0)), 40, -180 + i*30);
        ofDrawBitmapString(ofToString(hidden_output.coeffRef(i, 1)), 40, 0 + i*30);
        ofDrawBitmapString(ofToString(hidden_output.coeffRef(i, 2)), 40, 180 + i*30);
        
        // 出力層値
        ofDrawBitmapString(ofToString(output.coeffRef(i, 0)), 270, -180 + i*180);
        
        // 教師データ
        ofDrawBitmapString(ofToString(teacher.coeffRef(i, 0)), 360, -180 + i*180);
        
        // 出力値の誤差
        ofDrawBitmapString(ofToString(output_error.coeffRef(i, 0)), 300, -150 + (i*150 + i*30));
        // 隠れ層値の誤差
        ofDrawBitmapString(ofToString(hidden_error.coeffRef(i, 0)), -60, -150 + (i*150 + i*30));
    }
    
    
    ofPopMatrix();
    
    
    
    
    
    
    
    
    
    
    
    // Draw log
    //ofSetColor(255);
    //string info = ofToString(mat(3, 1)) + "\n";
    //info += "Particle Count: " + ofToString() + "\n";
    //ofDrawBitmapString(info, 30, 30);
}

void ofApp::keyPressed(int key){
    
}

void ofApp::keyReleased(int key){
    
}

void ofApp::mouseMoved(int x, int y ){
    
}

void ofApp::mouseDragged(int x, int y, int button){
    
}

void ofApp::mousePressed(int x, int y, int button){
    
}

void ofApp::mouseReleased(int x, int y, int button){
    
}

void ofApp::mouseEntered(int x, int y){
    
}

void ofApp::mouseExited(int x, int y){
    
}

void ofApp::windowResized(int w, int h){
    
}

void ofApp::gotMessage(ofMessage msg){
    
}

void ofApp::dragEvent(ofDragInfo dragInfo){
    
}
