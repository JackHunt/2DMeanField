#import "demo_app.h"

/*
 *Much of the code in this file is derived from http://www.philkr.net/code from the code
 *package for the paper "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials",
 *as such, all rights belong to Philip Krahenbuhl.
 */

using namespace MeanFieldDemo;

int colourToIndex(const cv::Vec3b &colour){
	return colour[0] + 256*colour[1] + 256*256*colour[2] + 1;
}

cv::Vec3b indexToColour(int idx){
	idx--;
	return cv::Vec3b(idx&0xff, (idx>>8)&0xff, (idx>>16)&0xff);
}

void unariesFromLabelling(const int *inputLabelling, float *outputUnary, int W, int H, int dim){
    const float GT_PROB = 0.7;
    const float u_energy = log( 1.0 / dim );
	const float n_energy = log( (1.0 - GT_PROB) / (dim-1) );
	const float p_energy = log( GT_PROB );
    
    for(int i=0; i<W*H; i++){
        if(inputLabelling[i] >= 0) {
            for(int k=0; k<dim; k++){
				outputUnary[i*dim + k] = n_energy;
			}
            outputUnary[i*dim + inputLabelling[i]] = p_energy;
        }else{
            for(int k=0; k<dim; k++){
				outputUnary[i*dim + k] = u_energy;
			}
        }
    }
}

void labellingFromUnaries(const float *inputUnaries, int *outputLabelling, int W, int H, int dim){
	
	for(int y=0; y<H; y++){
		for(int x=0; x<W; x++){
			int maxIdx = 0;
			float max = -10e+10;
			for(int k=0; k<dim; k++){
				if(inputUnaries[(y*W + x)*dim + k] > max){
					max = inputUnaries[(y*W + x)*dim + k];
					maxIdx = k;
				}
			}
			outputLabelling[y*W + x] = maxIdx;
		}
	}
}

void labellingToImage(cv::Mat &outputImage, const int *labelling, int W, int H){
	for(int y=0; y<H; y++){
		for (int x=0; x<W; x++){
            int c = colourMap[labelling[y*W + x]];
            outputImage.at<cv::Vec3b>(y, x) = indexToColour(c);
        } 
	}
}

void readLabellingFromImage(int *outputLabelling, const cv::Mat &inputImage, int W, int H, int dim){
    for(int y=0; y<H ;y++){
        for (int x=0; x<W; x++)
        {
            int c = colourToIndex(inputImage.at<cv::Vec3b>(y,x));

			int i;
			for(i=0; i<noColours && c!=colourMap[i]; i++);

			if(i == noColours){
                if(i<dim){
					colourMap[noColours++] = c;
				}else{
					c = 0;
				}
            }
            outputLabelling[y*W + x] = c ? i : -1;
        }
	}
}

int main(int argc, char* argv[]){
	//Check we have enough arguments.
	if(argc < 3){
		std::cout << "USAGE: <Input Image> <Rough Annotations>" << std::endl;
		return -1;
	}

	//Read in RGB image and annotation.
	cv::Mat rgbImage = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	cv::Mat annotImage = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
	
	if(!rgbImage.data || !annotImage.data){
		std::cout << "Error reading one or more images." << std::endl;
		return -1;
	}

	//Allocate storge.
	float *unaries = new float[annotImage.cols*annotImage.rows*M];
	int *annotLabelling = new int[annotImage.cols*annotImage.rows];
	int *outputLabelling = new int[annotImage.cols*annotImage.rows];
	
	//Get unaries from provided labelling.
	readLabellingFromImage(annotLabelling, annotImage, annotImage.cols, annotImage.rows, M);
	unariesFromLabelling(annotLabelling, unaries, annotImage.cols, annotImage.rows, M);

	//Create a new CRF and perform Mean Field Inference.
	MeanField::CPU::CRF crf(annotImage.cols, annotImage.rows, M, 10.0, 30.0, 15.0);
	crf.setSpatialWeight(5.0);
	crf.setBilateralWeight(10.0);
	crf.runInference(rgbImage.data, unaries, 5);

	//Get updated labelling from MAP estimate.
	float *outputUnaries = crf.getQ();
	labellingFromUnaries(outputUnaries, outputLabelling, annotImage.cols, annotImage.rows, M);

	//Build output image.
	cv::Mat outputImage(annotImage.rows, annotImage.cols, CV_8UC3);
	labellingToImage(outputImage, outputLabelling, annotImage.cols, annotImage.rows);

	//Display images.
	cv::namedWindow("Original RGB Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original RGB Image", rgbImage);
	cv::namedWindow("Original Annotations", cv::WINDOW_AUTOSIZE);
	cv::imshow("Original Annotations", annotImage);
	cv::namedWindow("Mean Field Output", cv::WINDOW_AUTOSIZE);
	cv::imshow("Mean Field Output", outputImage);
	cv::waitKey(0);

	//Clean up.
	delete[] unaries;
	delete[] annotLabelling;
	delete[] outputLabelling;
}
