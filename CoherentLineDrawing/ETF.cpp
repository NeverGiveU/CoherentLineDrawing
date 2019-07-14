#include "ETF.h"
#define M_PI 3.14159265358979323846;

using namespace cv;

ETF::ETF() {
	Size s(300, 300);
	Init(s);
}

ETF::ETF(Size s) {
	Init(s);
}

void ETF::Init(Size s) {
	this->flowField = Mat::zeros(s, CV_32FC3);
	this->refinedETF = Mat::zeros(s, CV_32FC3);
	this->gradientMag= Mat::zeros(s, CV_32FC3);
}

/**
 * Generate initial ETF
 * by taking perpendicular vectors(counter-clockwise) from gradient map
 */
void ETF::initial_ETF(string file, Size s) {
	resizeMat(s);

	Mat src = imread(file, 1);
	Mat src_n;
	Mat grad;
	normalize(src, src_n, 0.0, 1.0, NORM_MINMAX, CV_32FC1);
    ///Read into 'src' and assign the normalized version to 'src_n'
	// GaussianBlur(src_n, src_n, Size(51, 51), 0, 0);

	///Generate grad_x and grad_y
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
	Sobel(src_n, grad_x, CV_32FC1, 1, 0, 5);
	Sobel(src_n, grad_y, CV_32FC1, 0, 1, 5);

	///Compute gradient
	magnitude(grad_x, grad_y, this->gradientMag);
	normalize(this->gradientMag, this->gradientMag, 0.0, 1.0, NORM_MINMAX);

	this->flowField = Mat::zeros(src.size(), CV_32FC3);

#pragma omp parallel for
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; j++) {
			Vec3f u = grad_x.at<Vec3f>(i, j);
			Vec3f v = grad_y.at<Vec3f>(i, j);

			this->flowField.at<Vec3f>(i, j) = normalize(Vec3f(v.val[0], u.val[0], 0));
		}
	}
	this->rotateFlow(this->flowField, this->flowField, 90);
}

void ETF::refine_ETF(int kernel) {
#pragma omp parallel for
	for (int r = 0; r < this->flowField.rows; r++) {
		for (int c = 0; c < this->flowField.cols; ++c) {
			computeNewVector(c, r, kernel);
		}
	}
	this->flowField = this->refinedETF.clone();
}

/*
 * Ed(1)
 */
void ETF::computeNewVector(int x, int y, const int kernel) {
	const Vec3f t_cur_x = this->flowField.at<Vec3f>(y, x);
	Vec3f t_new = Vec3f(0, 0, 0);

	for (int r = y - kernel; r <= y + kernel; r++) {
		for (int c = x - kernel; c <= x + kernel; c++) {
			if (r < 0 || r >= this->refinedETF.rows || c < 0 || c >= this->refinedETF.cols) continue;

			const Vec3f t_cur_y = flowField.at<Vec3f>(r, c);
			float phi = computePhi(t_cur_x, t_cur_y);
			float w_s = computeWs(Point2f(x, y), Point2f(c, r), kernel);
			float w_m = computeWm(norm(gradientMag.at<Vec3f>(y, x)), norm(gradientMag.at<float>(r, c)));
			float w_d = computeWd(t_cur_x, t_cur_y);
			t_new += phi * t_cur_y*w_s*w_m*w_d;
		}
	}
	this->refinedETF.at<Vec3f>(y, x) = normalize(t_new);
}

/*
 * Eq (5)
 */
float ETF::computePhi(cv::Vec3f x, cv::Vec3f y) {
	return x.dot(y) > 0 ? 1 : -1;
}

/*
 * Eq (2)
 */
float ETF::computeWs(cv::Point2f x, Point2f y, int r) {
	return norm(x - y) < r ? 1 : 0;
}

/*
 * Eq (3)
 */
float ETF::computeWm(float gradmag_x, float gradmag_y) {
	float wm = (1 + tanh(gradmag_y - gradmag_x)) / 2;
	return wm;
}

/*
 * Eq (4)
 */
float ETF::computeWd(cv::Vec3f x, cv::Vec3f y) {
	return abs(x.dot(y));
}

void ETF::resizeMat(Size s) {
	resize(this->flowField, this->flowField, s, 0, 0, CV_INTER_LINEAR);
	resize(this->refinedETF, this->refinedETF, s, 0, 0, CV_INTER_LINEAR);
	resize(this->gradientMag, this->gradientMag, s, 0, 0, CV_INTER_LINEAR);
}

void ETF::rotateFlow(Mat& src, Mat& dst, float theta) {
	theta = theta / 180.0 * M_PI;
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			Vec3f v = src.at<cv::Vec3f>(i, j);
			float rx = v[0] * cos(theta) - v[1] * sin(theta);
			float ry = v[1] * cos(theta) + v[0] * sin(theta);
			dst.at<cv::Vec3f>(i, j) = Vec3f(rx, ry, 0.0);
		}
	}
}