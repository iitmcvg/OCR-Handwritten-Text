#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include<fstream>
#include<iostream>
#include<stdlib.h>
using namespace std;
using namespace cv;
struct data
{
  int img[28][28];
} s1;
int main()
{
  ifstream f("segments.dat");
  Mat m1(28,28,CV_8UC1);
while(f.read((char*)&s1,sizeof(s1)))
{
for(int x=0;x<28;x++)
for(int y=0;y<28;y++)
m1.at<uchar>(y,x)=s1.img[y][x];
imshow("",m1);
waitKey(0);
}
f.close();
return 0;
}
