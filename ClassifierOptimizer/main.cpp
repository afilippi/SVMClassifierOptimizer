#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <stdlib.h> 
#include <stdio.h>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include <conio.h>
#include <time.h>
#include <dos.h>
#include <windows.h>
#include <tchar.h>
#include <stdlib.h>   

#include <chrono>

#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

enum Dependent { POSITIVE, NEGATIVE};
int img_area = 250 * 125;


void print(vector<String> images, Dependent dept)
{
	for (int p = 1; p < images.size(); p++)
	{
		Mat input_mat = imread(images[p], 0);
		String base = "NEWNEGATIVES\\";
		if (dept == POSITIVE)
		{
			base = "NEWPOSITIVES\\";
		}
		imwrite(base + std::to_string(p) + ".jpg" , input_mat);
	}
}

std::vector<int> getGroundTruth(int numberOfPositives, int numberOfNegatives)
{
	std::vector<int> labels;

	for (int m = 0; m < numberOfPositives + numberOfNegatives; m++)
	{
		if (m < numberOfPositives)
		{
			labels.push_back(1);
		}
		else
		{
			labels.push_back(-1);
		}
	}


	return labels;
}


void loadTrainingMat(Mat& training_mat, Mat img_mat,int rowNum)
{
	int ii = 0; // Current column in training_mat
	for (int i = 0; i < img_mat.rows; i++) {
		for (int j = 0; j < img_mat.cols; j++) {
			training_mat.at<float>(rowNum, ii++) = img_mat.at<uchar>(i, j);

		}
	}
}

Mat stringify(Mat img_mat)
{
	int img_area = 250 * 125;
	Mat training_mat(1, img_area, CV_32FC1);

	int ii = 0; // Current column in training_mat
	for (int i = 0; i < img_mat.rows; i++)
	{
		for (int j = 0; j < img_mat.cols; j++)
		{
			training_mat.at<float>(0, ii++) = img_mat.at<uchar>(i, j);
		}
	}
	return training_mat;
}

int testImages(vector<String> imageSet, Ptr<SVM> svm)
{
	int numberOfHits = 0;
	for (int p = 1; p < imageSet.size(); p++)
	{
		Mat input_mat = imread(imageSet[p], 0);
		Mat training_mat = stringify(input_mat);

		float val1 = svm->predict(training_mat);

		bool detected = val1 >= 0;

		if (detected)
		{
			numberOfHits++;
		}
	}

	return numberOfHits;
}

float runClassifierTest(Ptr<SVM> svm, vector<String> masterPositiveFileNames, vector<String> masterNegativeFileNames)
{
	float falsePositive = 0;
	float trueNegative = 0;
	float truePositive = 0;
	float falseNegative = 0;

	float testPositives = masterPositiveFileNames.size();
	float testNegatives = masterNegativeFileNames.size();

	truePositive = testImages(masterPositiveFileNames, svm);
	falsePositive = testImages(masterNegativeFileNames, svm);

	trueNegative = testNegatives - falsePositive;
	falseNegative = testPositives - truePositive;

	float posRat = (truePositive / testPositives) * 100;
	float negRat = (trueNegative / testNegatives) * 100;
	float total = negRat + posRat;

	std::printf("Positve Rate %f \n", posRat);
	std::printf("Negative Rate %f \n", negRat);
	std::printf("Total Rate %f \n", total);
	std::printf("True Positive %f False Negative %f \n", truePositive, falseNegative);
	std::printf("True Negative %f False Positive %f \n\n", trueNegative, falsePositive);

	return total;
}


Ptr<SVM> runClassifierTraining(std::vector<String> masterPositiveFileNames, std::vector<String> masterNegativeFileNames, SVM::Params params)
{

	Ptr<SVM> svm;
	
	///start train classifier
	int numberOfPositives = masterPositiveFileNames.size();
	int numberOfNegatives = masterNegativeFileNames.size();
	int fileNum = numberOfPositives + numberOfNegatives;


	Mat training_mat(fileNum, img_area, CV_32FC1);

	int currentFileNumber = 0;

	for (int i = 0; i < numberOfPositives; i++)
	{
		Mat input_mat = imread(masterPositiveFileNames[i], 0);
		loadTrainingMat(training_mat, input_mat, currentFileNumber);
		currentFileNumber++;
	}

	for (int i = 0; i < numberOfNegatives; i++)
	{
		Mat input_mat = imread(masterNegativeFileNames[i], 0);
		loadTrainingMat(training_mat, input_mat, currentFileNumber);
		currentFileNumber++;
	}

	vector<int> labels;
	
	labels = getGroundTruth(numberOfPositives, numberOfNegatives);

	Mat labelsMat(numberOfPositives + numberOfNegatives, 1, CV_32SC1, &labels[0]);

	svm = StatModel::train<SVM>(training_mat, ROW_SAMPLE, labelsMat, params);	

	return svm;
}



void optimizeClassifier(String posBase, String negBase, int numberOfPositives, int numberOfNegatives, int chunkSize, Dependent dept, SVM::Params params)
{


	std::vector<String> masterPositiveFileNames;
	std::vector<String> masterNegativeFileNames;

	std::vector<String> masterNegativeTestFileNames;
	std::vector<String> masterPositiveTestFileNames;

	Ptr<SVM> svm;

	for (int i = 1; i < numberOfPositives + 1; i++)
	{
		std::string s = std::to_string(i);
		masterPositiveFileNames.push_back(posBase + "pos (" + s + ").jpg");
	}


	for (int i = 1; i < numberOfNegatives + 1; i++)
	{
		std::string s = std::to_string(i);
		masterNegativeFileNames.push_back(negBase + "neg (" + s + ").jpg");
	}

	masterNegativeTestFileNames = masterNegativeFileNames;
	masterPositiveTestFileNames = masterPositiveFileNames;

	svm = runClassifierTraining(masterPositiveFileNames, masterNegativeFileNames, params);
	float bestTotal = runClassifierTest(svm, masterPositiveTestFileNames, masterNegativeTestFileNames);

	int numberOfTestImages = numberOfNegatives;
	if (dept == POSITIVE)
	{
		numberOfTestImages = numberOfPositives;
	}

	int chunkIterator = 0;

	for (int i = 0; i < numberOfTestImages; i +=chunkSize)
	{
		vector<String> negatives = masterNegativeFileNames;
		vector<String> positives = masterPositiveFileNames;

		int currentNumberOfTestImages = masterNegativeFileNames.size();
		if (dept == POSITIVE)
		{
			currentNumberOfTestImages = masterPositiveFileNames.size();
		}

		int processedLength = chunkSize;

		if (processedLength + chunkIterator > currentNumberOfTestImages)
		{
			processedLength = currentNumberOfTestImages - chunkIterator;
		}

		if (dept == NEGATIVE)
		{
			negatives.erase(negatives.begin() + chunkIterator, negatives.begin() + chunkIterator + processedLength);
		}
		else
		{
			positives.erase(negatives.begin() + chunkIterator, negatives.begin() + chunkIterator + processedLength);
		}

		svm = runClassifierTraining(positives, negatives, params);

		float total = runClassifierTest(svm, masterPositiveTestFileNames, masterNegativeTestFileNames);

		if (total > bestTotal)
		{
			bestTotal = total;
			svm->save(std::to_string(bestTotal)  +"motionHistory.xml");

			if (dept == NEGATIVE)
			{
				masterNegativeFileNames = negatives;
			}
			else
			{
				masterPositiveFileNames = positives;
			}
			
		}
		else
		{
			chunkIterator += chunkSize;
			
		}

	}
	
	std::printf("/////FINISHED/////// \n\n");

	svm = runClassifierTraining(masterPositiveFileNames, masterNegativeFileNames, params);

	
	float finalTotal = runClassifierTest(svm, masterPositiveTestFileNames, masterNegativeTestFileNames);

	if (dept == NEGATIVE)
	{
		print(masterNegativeFileNames, NEGATIVE);
	}
	else
	{
		print(masterPositiveFileNames, POSITIVE);
	}
	
}



int main(int, char)
{
	string posBase = "C:\\tests\\ClassInSession\\MotionHistory\\Positive15\\";
	string negBase = "C:\\tests\\ClassInSession\\MotionHistory\\Negative15\\";

	Dependent dept = NEGATIVE;

	int chunkSize = 100;

	int numberOfPositives = 464;
	int numberOfNegatives = 533;

	SVM::Params params;
	params.svmType = SVM::C_SVC;
	params.kernelType = SVM::LINEAR;
	params.termCrit = TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6);

	optimizeClassifier(posBase, negBase, numberOfPositives, numberOfNegatives, chunkSize, dept, params);

	return 0;
}