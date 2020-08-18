#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat pseudoInverse(Mat src){

  int rows = src.rows;
  int columns = src.cols;
  
  //Single Value Decomposition
  Mat S, U, Vt;
  SVDecomp(src, S, U, Vt, SVD::FULL_UV);
  
  double max;
	minMaxIdx(S, 0, &max);

	double cutoff = pow(10, -15) * max;
	threshold(S, S, cutoff, 0, THRESH_TOZERO);

  //S+, transpose and element-wise reciprocal of S
  Mat diagonalS = Mat::zeros(rows, columns, CV_32FC1);
  for (int i = 0; i < S.rows; i++) {
    diagonalS.at<float>(i, i) = 1/S.at<float>(i);
  }

  //V* = Vt therefore A+ = V x S+ x Ut
  Mat pseudo = Vt.t() * diagonalS.t() * U.t();

  return pseudo;
}

int main(){

  Mat test = (Mat_<float>(2, 6) << 1,2,3,4,5,6,7,8,9,10,11,12);
  Mat result = pseudoInverse(test);
  
  cout << result;
  
  return 0;
}
