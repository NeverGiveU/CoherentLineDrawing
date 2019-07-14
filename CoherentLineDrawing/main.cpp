#include "CLD.h"
#include "PostProcessing.h"

using namespace cv;
using namespace std;

int main() {
	int ETF_kernel = 5;
	int ETF_iteration = 0;
	int FDoG_iteration = 0;

	CLD cld = CLD();

    ///Testing sample
	string path = "C:/Users/marry/Desktop/小黑要认真一些/十分钟尝试/相干线边缘提取/Coherent-Line-Drawing/data/eagle2.jpg"; 
	// specify your own image path ↑

	Mat img = imread(path);
	// cout << "The img size is: " << img.cols << " x " << img.rows << " !" << endl;
	Size s = Size(img.cols, img.rows);
	// cout << s << endl;

	cld.init(s);
	cld.readSrc(path);
	cld.genCLD();
	// cld.etf.refine_ETF(ETF_kernel);
	// cout << cld.result.cols << " x "<< cld.result.rows << endl;

	namedWindow("Image");
	imshow("Image", cld.result);
	waitKey(100000);
	return 0;

}