// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include <chrono>
#include <thread>

unsigned int noThreads = 1;

wchar_t* projectPath;

using namespace std;

float cosineValuesDCT[64];
float cosineValuesInverseDCT[64];

struct quantizationMatricesStruct {
	int compressionFactor;
	int matrix[8][8];
};

vector<vector<pair<char, char>>>finalLists;

struct BlockThreadArgsCompress {
	int qualityFactor;
	vector<Mat_<float>> batchOfBlocks;
	int batchIndex;
	int blocksNumberInBatch;
};

struct BlockThreadArgsDecompress {
	int qualityFactor;
	vector<Mat_<float>> batchOfBlocks;
	int batchIndex;
	int blocksNumberInBatch;
	int blocksPerRow;
	Mat_<Vec3b> result;
	int channel;
};

int numberOfBlocksPerThread;

vector<quantizationMatricesStruct> quantizationMatrices;

int baseQuantizationMatrix[8][8] = {
	{16, 11, 10, 16, 24, 40, 51, 61},
	{12, 12, 14, 19, 26, 58, 60, 55},
	{14, 13, 16, 24, 40, 57, 69, 56},
	{14, 17, 22, 29, 51, 87, 20, 62},
	{18, 22, 37, 56, 68, 109, 103, 77},
	{24, 35, 55, 64, 81, 104, 113, 92},
	{49, 64, 78, 87, 103, 121, 120, 101},
	{72, 92, 95, 98, 112, 100, 103, 99} };

void initQuantizationMatrices()
{
	for (int q = 10; q <= 100; q+=10)
	{
		quantizationMatricesStruct quantizationMatrix;
		quantizationMatrix.compressionFactor = q;

		int newMatrix[8][8];
		float S = 0;

		if (q < 50)
		{
			S = 5000.0f / q;
		}
		else
		{
			S = 200 - 2 * q;
		}

		for (int i = 0; i < 8; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				newMatrix[i][j] = floor((float)(S * baseQuantizationMatrix[i][j] + 50) / 100);
				if (newMatrix[i][j] == 0)
				{
					newMatrix[i][j] = 1;
				}
			}
		}

		memcpy(quantizationMatrix.matrix, newMatrix, sizeof(newMatrix));
		quantizationMatrices.push_back(quantizationMatrix);
	}
}

void getNumberOfThreads()
{
	noThreads = thread::hardware_concurrency();
	if (noThreads == 0)
	{
		printf("Cannot get the number of threads");
		exit(1);
	}
}

void initCosineValues()
{
	float pi = atan(1) * 4;
	float alfaCst = 0.7071067f;
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			cosineValuesDCT[i * 8 + j] = cos(((2 * i + 1) * j * pi) / 16);
			cosineValuesInverseDCT[i * 8 + j] = cosineValuesDCT[i * 8 + j];
			if (j == 0)
			{
				cosineValuesInverseDCT[i * 8 + j] *= alfaCst;
			}
		}
	}
}


template<typename T>
void doActualPadding(Mat_<T> src, Mat_<T> padded)
{
	int i, j;

	//the normal part
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			padded[i][j] = src[i][j];
		}
	}

	//left part
	for (i = 0; i < src.rows; i++)
	{
		for (j = src.cols; j < padded.cols; j++)
		{
			padded[i][j] = src[i][src.cols - 1];
		}
	}

	//bottom part
	for (i = src.rows; i < padded.rows; i++)
	{
		for (j = 0; j < src.cols; j++)
		{
			padded[i][j] = src[src.rows - 1][j];
		}
	}

	//left bottom part
	//first idea copy what was padded in right column

	for (i = src.rows; i < padded.rows; i++)
	{
		for (j = src.cols; j < padded.cols; j++)
		{
			padded[i][j] = src[src.rows - 1][j];
		}
	}
}

template<typename T>

Mat_<T> padMatrix(Mat_<T> src, char multiple)
{
	Mat_<T> paddedImage;
	if (src.rows % multiple == 0 && src.cols % multiple == 0)
		return src;
	else if (src.cols % multiple == 0)
	{
		paddedImage = Mat_<T>(src.rows + (multiple - (src.rows % multiple)), src.cols);
	}
	else if (src.rows % multiple == 0)
	{
		paddedImage = Mat_<T>(src.rows, src.cols + (multiple - (src.cols % multiple)));
	}
	else
	{
		paddedImage = Mat_<T>(src.rows + (multiple - (src.rows % multiple)), src.cols + (multiple - (src.cols % multiple)));
	}

	doActualPadding(src, paddedImage);
	return paddedImage;
}

Mat_<uchar> downSampeling_4_2_0(Mat_<Vec3b> src, char channelNumber)
{
	Mat_<uchar> downSampledChannel(src.rows / 2, src.cols / 2);
	for (int i = 0; i < src.rows; i += 2)
	{
		for (int j = 0; j < src.cols; j += 2)
		{
			downSampledChannel(i / 2, j / 2) = (src(i, j)[channelNumber] + src(i + 1, j)[channelNumber] + src(i, j + 1)[channelNumber] + src(i + 1, j + 1)[channelNumber]) / 4;
		}
	}

	return downSampledChannel;
}

void upSampleLinearInterpolating(Mat_<Vec3b> matrix, int channel, Mat_<float> block, int i, int j)
{
	for (int x = 0; x < 7; x++)
	{
		for (int y = 0; y < 7; y++)
		{
			if (i * 16 + x * 2 < matrix.rows && j * 16 + y * 2 < matrix.cols)
			{
				matrix(i * 16 + x * 2, j * 16 + y * 2)[channel] = (char)block(x, y);
			}
			if (i * 16 + x * 2 + 1 < matrix.rows && j * 16 + y * 2 < matrix.cols)
			{
				matrix(i * 16 + x * 2 + 1, j * 16 + y * 2)[channel] = ((int)block(x, y) + (int)block(x + 1, y)) / 2;
			}
			if (i * 16 + x * 2 < matrix.rows && j * 16 + y * 2 + 1 < matrix.cols)
			{
				matrix(i * 16 + x * 2, j * 16 + y * 2 + 1)[channel] = ((int)block(x, y) + (int)block(x, y + 1)) / 2;
			}
			if (i * 16 + x * 2 + 1 < matrix.rows && j * 16 + y * 2 + 1 < matrix.cols)
			{
				matrix(i * 16 + x * 2 + 1, j * 16 + y * 2 + 1)[channel] = ((int)block(x, y) + (int)block(x + 1, y + 1)) / 2;
			}
		}
	}

	for (int x = 0; x < 8; x++)
	{
		if (i * 16 + x * 2 < matrix.rows && j * 16 + 7 * 2 < matrix.cols)
		{
			matrix(i * 16 + x * 2, j * 16 + 7 * 2)[channel] = (char)block(x, 7);
		}
		if (i * 16 + x * 2 + 1 < matrix.rows && j * 16 + 7 * 2 < matrix.cols)
		{
			matrix(i * 16 + x * 2 + 1, j * 16 + 7 * 2)[channel] = (char)block(x, 7);
		}
		if (i * 16 + x * 2 < matrix.rows && j * 16 + 7 * 2 + 1 < matrix.cols)
		{
			matrix(i * 16 + x * 2, j * 16 + 7 * 2 + 1)[channel] = (char)block(x, 7);
		}
		if (i * 16 + x * 2 + 1 < matrix.rows && j * 16 + 7 * 2 + 1 < matrix.cols)
		{
			matrix(i * 16 + x * 2 + 1, j * 16 + 7 * 2 + 1)[channel] = (char)block(x, 7);
		}

		if (i * 16 + 7 * 2 < matrix.rows && j * 16 + x * 2 < matrix.cols)
		{
			matrix(i * 16 + 7 * 2, j * 16 + x * 2)[channel] = (char)block(7, x);
		}
		if (i * 16 + 7 * 2 + 1 < matrix.rows && j * 16 + x * 2 < matrix.cols)
		{
			matrix(i * 16 + 7 * 2 + 1, j * 16 + x * 2)[channel] = (char)block(7, x);
		}
		if (i * 16 + 7 * 2 < matrix.rows && j * 16 + x * 2 + 1 < matrix.cols)
		{
			matrix(i * 16 + 7 * 2, j * 16 + x * 2 + 1)[channel] = (char)block(7, x);
		}
		if (i * 16 + 7 * 2 + 1 < matrix.rows && j * 16 + x * 2 + 1 < matrix.cols)
		{
			matrix(i * 16 + 7 * 2 + 1, j * 16 + x * 2 + 1)[channel] = (char)block(7, x);
		}
	}

}

void upSample(Mat_<Vec3b> matrix, int channel, Mat_<float> block, int i, int j)
{
	for (int x = 0; x < 8; x++)
	{
		for (int y = 0; y < 8; y++)
		{
			matrix(i * 16 + x * 2, j * 16 + y * 2)[channel] = (char)block(x, y);
			matrix(i * 16 + x * 2 + 1, j * 16 + y * 2)[channel] = (char)block(x, y);
			matrix(i * 16 + x * 2, j * 16 + y * 2 + 1)[channel] = (char)block(x, y);
			matrix(i * 16 + x * 2 + 1, j * 16 + y * 2 + 1)[channel] = (char)block(x, y);
		}
	}

}

void mapBlockToRange(Mat_<float> block)
{
	for (int i = 0; i < block.rows; i++)
	{
		for (int j = 0; j < block.cols; j++)
		{
			block(i, j) -= 128;
		}
	}
}

void demapBlockToRange(Mat_<float> block)
{
	for (int i = 0; i < block.rows; i++)
	{
		for (int j = 0; j < block.cols; j++)
		{
			block(i, j) += 128;
			if (block(i, j) < 0)
			{
				block(i, j) = 0;
			}
			else if (block(i, j) > 255)
			{
				block(i, j) = 255;
			}
		}
	}
}

template<typename T>
void applyDCT(Mat_<T> &block)
{
	Mat_<float> result(8, 8);

	for (int u = 0; u < 8; u++)
	{
		for (int v = 0; v < 8; v++)
		{
			float term = 0.25f;
			if (u == 0 && v == 0)
			{
				term = 0.12499997f;
			}
			else if (v == 0 || u == 0)
			{
				term = 0.176776675;
			}

			float sum = 0.0f;
			for (int x = 0; x < 8; x++)
			{
				for (int y = 0; y < 8; y++)
				{
					sum += block(x, y) * cosineValuesDCT[x * 8 + u] * cosineValuesDCT[y * 8 + v];
				}
			}
			result(u, v) = term * sum;
		}
	}

	block = result;
	result.release();
}

template<typename T>
void applyInverseDCT(Mat_<T> &block)
{
	Mat_<float> result(8, 8);

	for (int x = 0; x < 8; x++)
	{
		for (int y = 0; y < 8; y++)
		{
			float sum = 0.0f;
			float term = 0.25f;
			for (int u = 0; u < 8; u++)
			{
				for (int v = 0; v < 8; v++)
				{
					float term2 = block(u, v) * cosineValuesInverseDCT[x * 8 + u] * cosineValuesInverseDCT[y * 8 + v];
					sum += term2;
				}
			}
			result(x, y) = term * sum;
			result(x, y) = round(result(x, y));
		}
	}

	block = result;
	result.release();
}

template<typename T>
void compressBlockOf8(Mat_ <T> block, int factor)
{
	for (int i = 0; i < quantizationMatrices.size(); i++)
	{
		if (factor == quantizationMatrices[i].compressionFactor)
		{
			for (int x = 0; x < block.rows; x++)
			{
				for (int y = 0; y < block.cols; y++)
				{
					block(x, y) /= quantizationMatrices[i].matrix[x][y];
					block(x, y) = round(block(x, y));
				}
			}
			return;
		}
	}
}

template<typename T>
void decompressBlockOf8(Mat_ <T> block, int factor)
{
	for (int i = 0; i < quantizationMatrices.size(); i++)
	{
		if (factor == quantizationMatrices[i].compressionFactor)
		{
			for (int x = 0; x < block.rows; x++)
			{
				for (int y = 0; y < block.cols; y++)
				{
					block(x, y) *= quantizationMatrices[i].matrix[x][y];
				}
			}
			return;
		}
	}
}


vector<float> zigzag(Mat_<float> block)
{	//hardocded for 8x8 block
	vector<float> result;
	for (int j = 0; j < 4; j++)
	{
		for (int i = 2 * j; i >= 0; i--) // in sus
		{
			result.push_back(block(i, 2 * j - i));
		}
		for (int i = 2 * j + 1; i >= 0; i--) // in jos
		{
			result.push_back(block(2 * j + 1 - i, i));
		}
	}

	for (int j = 3; j > 0; j--)
	{
		for (int i = 2 * j + 1; i > 0; i--) // in sus
		{
			result.push_back(block(7 - 2 * j - 1 + i, 8 - i));
		}
		for (int i = 2 * j; i > 0; i--) // in jos
		{
			result.push_back(block(9 - i - 1, 7 - 2 * j + i));
		}
	}
	result.push_back(block(7, 7));

	return result;
}

Mat_<char> reverseZigZag(char* blockZigZagged)
{
	Mat_<char> block(8, 8);
	char indexInBlock = 0;
	for (int j = 0; j < 4; j++)
	{
		for (int i = 2 * j; i >= 0; i--) // in sus
		{
			block(i, 2 * j - i) = blockZigZagged[indexInBlock++];
		}
		for (int i = 2 * j + 1; i >= 0; i--) // in jos
		{
			block(2 * j + 1 - i, i) = blockZigZagged[indexInBlock++];
		}
	}

	for (int j = 3; j > 0; j--)
	{
		for (int i = 2 * j + 1; i > 0; i--) // in sus
		{
			block(7 - 2 * j - 1 + i, 8 - i) = blockZigZagged[indexInBlock++];
		}
		for (int i = 2 * j; i > 0; i--) // in jos
		{
			block(9 - i - 1, 7 - 2 * j + i) = blockZigZagged[indexInBlock++];
		}
	}
	block(7, 7) = blockZigZagged[indexInBlock];

	return block;

}

vector<pair<char, char>> runLengthEncoding(vector<float> listedBlock)
{
	int index = 0;
	vector<pair<char, char>> result;

	while (index < listedBlock.size())
	{
		int counter = 1;
		while (index < listedBlock.size() - 1)
		{
			if (listedBlock[index] == listedBlock[index + 1])
			{
				counter++;
				index++;
			}
			else
				break;
		}
		result.push_back(pair<char, char>(listedBlock[index], counter));
		index++;
	}

	return result;
}

template<typename T>
void printBlock(Mat_<T> block)
{
	for (int i = 0; i < block.rows; i++)
	{
		cout << "Line "<<i<<":      ";
		for (int j = 0; j < block.cols; j++)
		{
			cout << block(i, j) << " ";
		}
		cout << endl;
	}

	cout << endl << endl << endl;
}

void printStatus(char compressing, int percent)
{
	system("cls");
	if (compressing)
	{
		printf("Compressing... ");
	}
	else
	{
		printf("Decompressing... ");
	}
	for (int i = 0; i < percent / 10; i++)
		printf("%c%c%c", 219, 219, 219);
	for (int i = 0; i < (10 - percent / 10); i++)
		printf("   ");
	printf(" %d%%", percent);
}

void compressBlock(BlockThreadArgsCompress& args) {
	for (int i = 0; i < args.blocksNumberInBatch; i++)
	{
		mapBlockToRange(args.batchOfBlocks[i]);
		applyDCT(args.batchOfBlocks[i]);
		compressBlockOf8(args.batchOfBlocks[i], args.qualityFactor);
		vector<float> zigZagResult = zigzag(args.batchOfBlocks[i]);
		vector<pair<char, char>> runLengthEncodingResults = runLengthEncoding(zigZagResult);
		finalLists[args.batchIndex * numberOfBlocksPerThread + i] = runLengthEncodingResults;
	}
}

void compressRemainingBlocks(vector<Mat_<float>> batchOfBlocks, int numberOfBlocks, int batchIndex, int numberOfBlocksPerThread, int qualityFactor)
{
	for (int i = 0; i < numberOfBlocks; i++)
	{
		mapBlockToRange(batchOfBlocks[i]);
		applyDCT(batchOfBlocks[i]);
		compressBlockOf8(batchOfBlocks[i], qualityFactor);
		vector<float> zigZagResult = zigzag(batchOfBlocks[i]);
		vector<pair<char, char>> runLengthEncodingResults = runLengthEncoding(zigZagResult);
		finalLists[batchIndex * numberOfBlocksPerThread + i] = runLengthEncodingResults;
	}
}

void compressChannel(Mat_<uchar> channel, int nrBlockChannelRows, int nrBlockChannelCols, int qualityFactor, FILE* outFile)
{
	finalLists.resize(nrBlockChannelRows * nrBlockChannelCols);

	vector<thread> threads;
	numberOfBlocksPerThread = nrBlockChannelRows * nrBlockChannelCols / noThreads;

	vector<Mat_<float>> batchOfBlocks(numberOfBlocksPerThread);

	int batchIndex = 0;
	int numberOfBlocks = 0;

	for (int i = 0; i < channel.rows; i += 8)
	{
		for (int j = 0; j < channel.cols; j += 8)
		{
			Mat_<float> blockOf8(8, 8);
			for (int m = 0; m < 8; m++)
			{
				for (int n = 0; n < 8; n++)
				{
					blockOf8(m, n) = channel(i + m, j + n);
				}
			}

			batchOfBlocks[(i / 8 * nrBlockChannelCols + j / 8) % numberOfBlocksPerThread] = blockOf8;

			numberOfBlocks++;

			if ((i / 8 * nrBlockChannelCols + j / 8 + 1) % numberOfBlocksPerThread == 0)
			{
				BlockThreadArgsCompress args;
				args.qualityFactor = qualityFactor;
				args.batchOfBlocks = batchOfBlocks;
				args.batchIndex = batchIndex;
				args.blocksNumberInBatch = numberOfBlocks;
				batchIndex++;
				threads.emplace_back(compressBlock, args);
				numberOfBlocks = 0;
			}
		}
	}

	for (int i = 0; i < threads.size(); i++)
	{
		threads[i].join();
	}

	threads.clear();

	if (numberOfBlocks != 0)
	{
		compressRemainingBlocks(batchOfBlocks, numberOfBlocks, batchIndex, numberOfBlocksPerThread, qualityFactor);
	}

	for (int i = 0; i < nrBlockChannelRows * nrBlockChannelCols; i++)
	{
		for (int x = 0; x < finalLists[i].size(); x++)
		{
			fwrite(&(finalLists[i][x].first), sizeof(char), 1, outFile);
			fwrite(&(finalLists[i][x].second), sizeof(char), 1, outFile);
		}
		char c = 0;
		fwrite(&c, sizeof(char), 1, outFile);
		fwrite(&c, sizeof(char), 1, outFile);
	}

}

void compressImage()
{
	char fname[MAX_PATH];
	FILE* outFile;

	while (openFileDlg(fname))
	{
		auto start = std::chrono::high_resolution_clock::now();
		Mat_<Vec3b> readImage = imread(fname, IMREAD_COLOR);
		Mat_<Vec3b> src(readImage.rows, readImage.cols);

		//order Y  Cr  Cb
		cvtColor(readImage, src, 36, 0);

		//pad image
		src = padMatrix(src, 2);

		//get Y Channel
		Mat_<uchar> YChannel(src.rows, src.cols);
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				YChannel(i, j) = src(i, j)[0];
			}
		}

		//downsample Cr
		Mat_<uchar> CrChannel = downSampeling_4_2_0(src, 1);

		//downsample Cb
		Mat_<uchar> CbChannel = downSampeling_4_2_0(src, 2);

		YChannel = padMatrix(YChannel, 8);
		CrChannel = padMatrix(CrChannel, 8);
		CbChannel = padMatrix(CbChannel, 8);

		int nrBlocksYRows = YChannel.rows / 8;
		int nrBlocksYCols = YChannel.cols / 8;

		int nrBlocksCrCbRows = CrChannel.rows / 8;
		int nrBlocksCrCbCols = CrChannel.cols / 8;

		char luminanceQuality = 50;
		char chromanceQuality = 50;

		const char* compressedExtension = "_compressed.bin";
		char* fullFilePath = (char*)malloc(strlen(fname) + strlen(compressedExtension) + 1);
		strcpy(fullFilePath, fname);
		char* extensionLocation = strstr(fname, ".bmp");
		strcpy(fullFilePath + strlen(fullFilePath) - 4, compressedExtension);

		outFile = fopen(fullFilePath, "wb");

		fwrite(&(src.rows), sizeof(int), 1, outFile);
		fwrite(&(src.cols), sizeof(int), 1, outFile);

		fwrite(&nrBlocksYRows, sizeof(int), 1, outFile);
		fwrite(&nrBlocksYCols, sizeof(int), 1, outFile);

		fwrite(&nrBlocksCrCbRows, sizeof(int), 1, outFile);
		fwrite(&nrBlocksCrCbCols, sizeof(int), 1, outFile);

		fwrite(&luminanceQuality, sizeof(char), 1, outFile);
		fwrite(&chromanceQuality, sizeof(char), 1, outFile);

		printStatus(1, 10);

		compressChannel(YChannel, nrBlocksYRows, nrBlocksYCols, luminanceQuality, outFile);

		printStatus(1, 40);

		compressChannel(CrChannel, nrBlocksCrCbRows, nrBlocksCrCbCols, chromanceQuality, outFile);

		printStatus(1, 70);

		compressChannel(CbChannel, nrBlocksCrCbRows, nrBlocksCrCbCols, chromanceQuality, outFile);

		printStatus(1, 100);
		fclose(outFile);

		auto end = std::chrono::high_resolution_clock::now();
		auto i_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		cout << endl << "Time elapsed: " << (float)i_ms.count() / 1000 << "s";
	}
}

Mat_<Vec3b> cutImageToSize(Mat_<Vec3b> image, int rows, int cols)
{
	Mat_<Vec3b> result(rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			result(i, j) = image(i, j);
		}
	}

	return result;
}

void decompressBlockYChannel(BlockThreadArgsDecompress& args) {
	for (int i = 0; i < args.blocksNumberInBatch; i++)
	{
		decompressBlockOf8(args.batchOfBlocks[i], args.qualityFactor);
		applyInverseDCT(args.batchOfBlocks[i]);
		demapBlockToRange(args.batchOfBlocks[i]);

		for (int x = 0; x < 8; x++)
		{
			for (int y = 0; y < 8; y++)
			{
				args.result((args.batchIndex * numberOfBlocksPerThread + i) / args.blocksPerRow * 8 + x, (args.batchIndex * numberOfBlocksPerThread + i) % args.blocksPerRow * 8 + y)[0] = (char)args.batchOfBlocks[i](x, y);
			}
		}
	}
}

void decompressRemainingBlocksYChannel(vector<Mat_<float>> batchOfBlocks, int numberOfBlocks, int batchIndex, int blocksPerRow, int numberOfBlocksPerThread, int qualityFactor, Mat_<Vec3b> result)
{
	for (int i = 0; i < numberOfBlocks; i++)
	{
		decompressBlockOf8(batchOfBlocks[i], qualityFactor);
		applyInverseDCT(batchOfBlocks[i]);
		demapBlockToRange(batchOfBlocks[i]);

		for (int x = 0; x < 8; x++)
		{
			for (int y = 0; y < 8; y++)
			{
				result((batchIndex * numberOfBlocksPerThread + i) / blocksPerRow * 8 + x, (batchIndex * numberOfBlocksPerThread + i) % blocksPerRow * 8 + y)[0] = (char)batchOfBlocks[i](x, y);

			}
		}
	}
}

void decompressBlockCrCbChannel(BlockThreadArgsDecompress& args) {
	for (int i = 0; i < args.blocksNumberInBatch; i++)
	{
		decompressBlockOf8(args.batchOfBlocks[i], args.qualityFactor);
		applyInverseDCT(args.batchOfBlocks[i]);
		demapBlockToRange(args.batchOfBlocks[i]);

		upSampleLinearInterpolating(args.result, args.channel, args.batchOfBlocks[i], (args.batchIndex * numberOfBlocksPerThread + i) / args.blocksPerRow, (args.batchIndex * numberOfBlocksPerThread + i) % args.blocksPerRow);

	}
}

void decompressRemainingBlocksCrCbChannel(vector<Mat_<float>> batchOfBlocks, int numberOfBlocks, int batchIndex, int blocksPerRow, int numberOfBlocksPerThread, int qualityFactor, Mat_<Vec3b> result, int channel)
{
	for (int i = 0; i < numberOfBlocks; i++)
	{
		decompressBlockOf8(batchOfBlocks[i], qualityFactor);
		applyInverseDCT(batchOfBlocks[i]);
		demapBlockToRange(batchOfBlocks[i]);

		upSampleLinearInterpolating(result, channel, batchOfBlocks[i], (batchIndex * numberOfBlocksPerThread + i) / blocksPerRow, (batchIndex * numberOfBlocksPerThread + i) % blocksPerRow);

	}
}

void decompressChannel(Mat_<Vec3b> resultInYCrCb, int nrBlocksChannelRows, int nrBlocksChannelCols, int qualityFactor, FILE* inFile, int channel)
{
	vector<thread> threads;
	numberOfBlocksPerThread = nrBlocksChannelRows * nrBlocksChannelCols / noThreads;

	vector<Mat_<float>> batchOfBlocks(numberOfBlocksPerThread);

	int batchIndex = 0;
	int numberOfBlocks = 0;

	for (int i = 0; i < nrBlocksChannelRows; i++)
	{
		for (int j = 0; j < nrBlocksChannelCols; j++)
		{
			char value = 0;
			char aparitions = 0;
			char blockZigZaged[64] = { 0 };
			char indexInBlock = 0;
			do {
				fread(&value, sizeof(char), 1, inFile);
				fread(&aparitions, sizeof(char), 1, inFile);
				if (value == 0 && aparitions == 0)
				{
					break;
				}
				for (int x = 0; x < aparitions; x++)
				{
					blockZigZaged[indexInBlock++] = value;
				}

			} while (1);

			Mat_<float> block = reverseZigZag(blockZigZaged);

			batchOfBlocks[(i * nrBlocksChannelCols + j) % numberOfBlocksPerThread] = block;
			numberOfBlocks++;

			if ((i * nrBlocksChannelCols + j + 1) % numberOfBlocksPerThread == 0)
			{
				BlockThreadArgsDecompress args;
				args.qualityFactor = qualityFactor;
				args.batchOfBlocks = batchOfBlocks;
				args.batchIndex = batchIndex;
				args.blocksNumberInBatch = numberOfBlocks;
				args.result = resultInYCrCb;
				args.blocksPerRow = nrBlocksChannelCols;
				args.channel = channel;
				batchIndex++;
				if (channel == 0)
				{
					threads.emplace_back(decompressBlockYChannel, args);
				}
				else
				{
					threads.emplace_back(decompressBlockCrCbChannel, args);
				}
				numberOfBlocks = 0;

			}
		}
	}

	if (numberOfBlocks != 0)
	{
		if (channel == 0)
		{
			decompressRemainingBlocksYChannel(batchOfBlocks, numberOfBlocks, batchIndex, nrBlocksChannelCols, numberOfBlocksPerThread, qualityFactor, resultInYCrCb);
		}
		else
		{
			decompressRemainingBlocksCrCbChannel(batchOfBlocks, numberOfBlocks, batchIndex, nrBlocksChannelCols, numberOfBlocksPerThread, qualityFactor, resultInYCrCb, channel);

		}
	}

	for (int i = 0; i < threads.size(); i++)
	{
		threads[i].join();
	}

	threads.clear();
}

void decompressImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		auto start = std::chrono::high_resolution_clock::now();

		FILE* inFile;

		inFile = fopen(fname, "rb");

		int originalSizeRows = 0;
		int originalSizeCols = 0;

		int nrBlocksYRows = 0;
		int nrBlocksYCols = 0;

		int nrBlocksCrCbRows = 0;
		int nrBlocksCrCbCols = 0;

		int luminanceQuality = 0;
		int chromanceQuality = 0;

		fread(&originalSizeRows, sizeof(int), 1, inFile);
		fread(&originalSizeCols, sizeof(int), 1, inFile);

		fread(&nrBlocksYRows, sizeof(int), 1, inFile);
		fread(&nrBlocksYCols, sizeof(int), 1, inFile);

		fread(&nrBlocksCrCbRows, sizeof(int), 1, inFile);
		fread(&nrBlocksCrCbCols, sizeof(int), 1, inFile);

		fread(&luminanceQuality, sizeof(char), 1, inFile);
		fread(&chromanceQuality, sizeof(char), 1, inFile);

		Mat_<Vec3b> resultInYCrCb(8 * nrBlocksYRows, 8 * nrBlocksYCols);
		Mat_<Vec3b> result(8 * nrBlocksYRows, 8 * nrBlocksYCols);

		printStatus(0, 10);

		decompressChannel(resultInYCrCb, nrBlocksYRows, nrBlocksYCols, luminanceQuality, inFile, 0);


		printStatus(0, 40);

		decompressChannel(resultInYCrCb, nrBlocksCrCbRows, nrBlocksCrCbCols, chromanceQuality, inFile, 1);
		

		printStatus(0, 70);
		
		decompressChannel(resultInYCrCb, nrBlocksCrCbRows, nrBlocksCrCbCols, chromanceQuality, inFile, 2);


		cvtColor(resultInYCrCb, result, 38, 0);

		//cut the image
		if ((nrBlocksYRows * 8 != originalSizeRows) || (nrBlocksYCols * 8 != originalSizeCols))
		{
			result = cutImageToSize(result, originalSizeRows, originalSizeCols);
		}

		printStatus(0, 100);

		const char* decompressedExtension = "_decompressed.bmp";
		//extra 2 chars for decompress instead of compress
		char* outputFileName = (char*)malloc(strlen(fname) + 2);

		strcpy(outputFileName, fname);
		strcpy(outputFileName + (strlen(fname) - 15), decompressedExtension);

		imwrite(outputFileName, result);
		auto end = std::chrono::high_resolution_clock::now();
		auto i_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		cout << endl << "Time elapsed: " << (float)i_ms.count() / 1000 << "s";
		imshow("decompressed Image", result);

		waitKey();
	}
}

void testZigZag()
{
	Mat_<float> test(8, 8);
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			test(i, j) = i * 8 + j;
		}
	}

	vector<float> zigZagged = zigzag(test);

	char array[64] = { 0 };
	for (int i = 0; i < 64; i++)
	{
		array[i] = (char)zigZagged[i];
	}

	Mat_<uchar> result = reverseZigZag(array);
	for (int i = 0; i < 8; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			cout << (int)test(i, j) << " ";
		}
		cout << endl;
	}
}

int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	getNumberOfThreads();
	initCosineValues();
	initQuantizationMatrices();

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Compress an image to JPEG\n");
		printf(" 2 - Decompress an image from JPEG\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			compressImage();
			break;
		case 2:
			decompressImage();
			break;
		default:
			return 0;
		}
	} while (op != 0);
	return 0;
}