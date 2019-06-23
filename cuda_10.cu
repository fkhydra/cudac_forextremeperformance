#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <windows.h>
#include <d2d1.h>
#include <d2d1helper.h>
#pragma comment(lib, "d2d1")
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")

//*****double buffering*****
#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1000

D2D1_RECT_U display_area;
ID2D1Bitmap *memkeptarolo = NULL;
unsigned int *dev_image_data, image_data[SCREEN_WIDTH * SCREEN_HEIGHT];
float *dev_zbuffer;//ez int is volt/lehet
typedef struct Vec3f {
	float x, y, z;
};
//**************************************

//**************PEGAZUS 3D************
#define MAX_OBJ_NUM 20000000
float zoo_value = 1.0;
int drawing_in_progress = 0;
int viewpoint = -500;
float persp_degree, current_zoom;
float rot_degree_x;
float rot_degree_y;
float rot_degree_z;
float rot_degree_x2 = 0;
float rot_degree_y2 = 90.0f;
float rot_degree_z2 = 0;
float Math_PI = 3.14159265358979323846;
float raw_verticesX[MAX_OBJ_NUM], raw_verticesY[MAX_OBJ_NUM], raw_verticesZ[MAX_OBJ_NUM];
int raw_vertex_counter;
int raw_vertices_length;
struct VEKTOR {
	float x;
	float y;
	float z;
};
VEKTOR Vector1, Vector2, vNormal;
//*******CUDA*************
float *dev_raw_verticesX, *dev_raw_verticesY, *dev_raw_verticesZ;
float *dev_rotated_verticesX, *dev_rotated_verticesY, *dev_rotated_verticesZ;
//************************
void init_3D(void);
void data_transfer_to_GPU(void);
void cleanup_matrices(void);
void D2D_drawing(void);
__global__ void CUDA_rotation(int maxitemcount, float *rawarrayX, float *rawarrayY, float *rawarrayZ, float *rotarrayX, float *rotarrayY, float *rotarrayZ, float degree_cosx, float degree_sinx, float degree_cosy, float degree_siny, float degree_cosz, float degree_sinz);
void drawing(void);
__global__ void render_objects(int maxitemcount, float *rotarrayX, float *rotarrayY, float *rotarrayZ, unsigned int *puffer, float *zpuffer);
__global__ void zoom_in(int maxitemcount, float *rawarrayX, float *rawarrayY, float *rawarrayZ);
__global__ void zoom_out(int maxitemcount, float *rawarrayX, float *rawarrayY, float *rawarrayZ);
//************************************

//***********STANDARD WIN32API WINDOWING************
ID2D1Factory* pD2DFactory = NULL;
ID2D1HwndRenderTarget* pRT = NULL;
#define HIBA_00 TEXT("Error:Program initialisation process.")
HINSTANCE hInstGlob;
int SajatiCmdShow;
char szClassName[] = "WindowsApp";
HWND Form1; //Ablak kezeloje
LRESULT CALLBACK WndProc0(HWND, UINT, WPARAM, LPARAM);
//******************************************************

//*******for measurements********
long int vertex_counter, poly_counter;
float fps_stat;
int starttime;
int endtime;

//*****double buffering*****
void create_main_buffer(void);
void CUDA_cleanup_main_buffer(void);
__global__ void CUDA_CleanUp_Zbuffer(float *zpuffer);
void swap_main_buffer(void);
//**************************************

//*****drawig algorithms*****
__device__ void CUDA_SetPixel(int x1, int y1, int color, unsigned int *puffer);
__device__ void CUDA_SetPixel_Zbuffer(int x1, int y1, int z1, int color, unsigned int *puffer, float *zpuffer);
__device__ void CUDA_DrawLine(int x1, int y1, int x2, int y2, int color, unsigned int *puffer);
__device__ void CUDA_DrawLine_Zbuffer(int x1, int y1, int z1, int x2, int y2, int z2, int color, unsigned int *puffer, float *zpuffer);
__device__ void CUDA_FillTriangle(int x1, int y1, int x2, int y2, int x3, int y3, int color, unsigned int *puffer);
__device__ void CUDA_FillTriangle_Zbuffer(int x1, int y1, int z1, int x2, int y2, int z2, int x3, int y3, int z3, int color, unsigned int *puffer, float *zpuffer);
//**************************************

//********************************
//OBJ format handling
//********************************
float tomb_vertices[MAX_OBJ_NUM][3];
int tomb_faces[MAX_OBJ_NUM][5];
int tomb_vertices_length = 0, tomb_faces_length = 0;
int getelementcount(unsigned char csv_content[]);
void getelement(unsigned char csv_content[], unsigned int data_index, unsigned char csv_content2[]);
void obj_loader(void);

//*********************************
//The main entry point of our program
//*********************************
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR szCmdLine, int iCmdShow)
{
	static TCHAR szAppName[] = TEXT("StdWinClassName");
	HWND hwnd;
	MSG msg;
	WNDCLASS wndclass0;
	SajatiCmdShow = iCmdShow;
	hInstGlob = hInstance;

	//*********************************
	//Preparing Windows class
	//*********************************
	wndclass0.style = CS_HREDRAW | CS_VREDRAW;
	wndclass0.lpfnWndProc = WndProc0;
	wndclass0.cbClsExtra = 0;
	wndclass0.cbWndExtra = 0;
	wndclass0.hInstance = hInstance;
	wndclass0.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndclass0.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndclass0.hbrBackground = (HBRUSH)GetStockObject(LTGRAY_BRUSH);
	wndclass0.lpszMenuName = NULL;
	wndclass0.lpszClassName = TEXT("WIN0");

	//*********************************
	//Registering our windows class
	//*********************************
	if (!RegisterClass(&wndclass0))
	{
		MessageBox(NULL, HIBA_00, TEXT("Program Start"), MB_ICONERROR);
		return 0;
	}

	//*********************************
	//Creating the window
	//*********************************
	Form1 = CreateWindow(TEXT("WIN0"),
		TEXT("CUDA - DIRECT2D"),
		(WS_OVERLAPPED | WS_SYSMENU | WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX),
		50,
		50,
		SCREEN_WIDTH,
		SCREEN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);

	//*********************************
	//Displaying the window
	//*********************************
	ShowWindow(Form1, SajatiCmdShow);
	UpdateWindow(Form1);

	//*********************************
	//Activating the message processing for our window
	//*********************************
	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	return msg.wParam;
}

//*********************************
//The window's callback funtcion: handling events
//*********************************
LRESULT CALLBACK WndProc0(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	HDC hdc;
	PAINTSTRUCT ps;
	unsigned int xPos, yPos, xPos2, yPos2, fwButtons;

	switch (message)
	{
	//*********************************
	//When creating the window
	//*********************************
	case WM_CREATE:
		D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &pD2DFactory);
		pD2DFactory->CreateHwndRenderTarget(
			D2D1::RenderTargetProperties(),
			D2D1::HwndRenderTargetProperties(
				hwnd, D2D1::SizeU(SCREEN_WIDTH, SCREEN_HEIGHT)),
			&pRT);
		create_main_buffer();
		cudaMalloc((void**)&dev_raw_verticesX, MAX_OBJ_NUM * sizeof(float));
		cudaMalloc((void**)&dev_raw_verticesY, MAX_OBJ_NUM * sizeof(float));
		cudaMalloc((void**)&dev_raw_verticesZ, MAX_OBJ_NUM * sizeof(float));
		cudaMalloc((void**)&dev_rotated_verticesX, MAX_OBJ_NUM * sizeof(float));
		cudaMalloc((void**)&dev_rotated_verticesY, MAX_OBJ_NUM * sizeof(float));
		cudaMalloc((void**)&dev_rotated_verticesZ, MAX_OBJ_NUM * sizeof(float));
		cudaMalloc((void**)&dev_image_data, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(unsigned int));
		cudaMalloc((void**)&dev_zbuffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(float));
		init_3D();
		obj_loader();
		data_transfer_to_GPU();
		if ((joyGetNumDevs()) > 0) joySetCapture(hwnd, JOYSTICKID1, NULL, FALSE);
		return 0;
	//*********************************
	//to eliminate color flickering
	//*********************************
	case WM_ERASEBKGND:
		return (LRESULT)1;
	case MM_JOY1MOVE:
		fwButtons = wParam;
		xPos = LOWORD(lParam);
		yPos = HIWORD(lParam);
		if (xPos == 65535) {
			rot_degree_y2 += 2.0; D2D_drawing();
		}
		else if (xPos == 0) {
			rot_degree_y2 -= 2.0; D2D_drawing();
		}
		if (yPos == 65535) {
			rot_degree_x2 += 2.0; D2D_drawing();
		}
		else if (yPos == 0) {
			rot_degree_x2 -= 2.0; D2D_drawing();
		}
		if (fwButtons == 128) {
			rot_degree_z2 += 2.0; D2D_drawing();
		}
		else if (fwButtons == 64) {
			rot_degree_z2 -= 2.0; D2D_drawing();
		}
		if (rot_degree_y2 > 360) {
			rot_degree_y2 = 0; D2D_drawing();
		}
		else if (rot_degree_y2 < 0) {
			rot_degree_y2 = 358; D2D_drawing();
		}
		if (rot_degree_x2 > 359) {
			rot_degree_x2 = 0; D2D_drawing();
		}
		else if (rot_degree_x2 < 0) {
			rot_degree_x2 = 358; D2D_drawing();
		}
		if (rot_degree_z2 > 359) {
			rot_degree_z2 = 0; D2D_drawing();
		}
		else if (rot_degree_z2 < 0) {
			rot_degree_z2 = 358; D2D_drawing();
		}

		if (fwButtons == 2)
		{
			int blockSize = 384;
			int numBlocks = (raw_vertices_length + blockSize - 1) / blockSize;
			zoo_value *= 1.02;
			zoom_in << <numBlocks, blockSize >> > (raw_vertices_length, dev_raw_verticesX, dev_raw_verticesY, dev_raw_verticesZ);
			cudaDeviceSynchronize();
			D2D_drawing();
		}
		else if (fwButtons == 4)
		{
			int blockSize = 384;
			int numBlocks = (raw_vertices_length + blockSize - 1) / blockSize;
			zoo_value /= 1.02;
			zoom_out << <numBlocks, blockSize >> > (raw_vertices_length, dev_raw_verticesX, dev_raw_verticesY, dev_raw_verticesZ);
			cudaDeviceSynchronize();
			D2D_drawing();
		}
		break;
	//*********************************
	//Repainting the client area of the window
	//*********************************
	case WM_PAINT:
		hdc = BeginPaint(hwnd, &ps);
		EndPaint(hwnd, &ps);
		D2D_drawing();
		return 0;
	//*********************************
	//Closing the window, freeing resources
	//*********************************
	case WM_CLOSE:
		pRT->Release();
		pD2DFactory->Release();
		cudaFree(dev_raw_verticesX);
		cudaFree(dev_raw_verticesY);
		cudaFree(dev_raw_verticesZ);
		cudaFree(dev_rotated_verticesX);
		cudaFree(dev_rotated_verticesY);
		cudaFree(dev_rotated_verticesZ);
		cudaFree(dev_image_data);
		cudaFree(dev_zbuffer);
		DestroyWindow(hwnd);
		return 0;
		//*********************************
		//Destroying the window
		//*********************************
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	}
	return DefWindowProc(hwnd, message, wParam, lParam);
}

//********************************
//PEGAZUS 3D
//********************************
void create_main_buffer(void)
{
	pRT->CreateBitmap(D2D1::SizeU(SCREEN_WIDTH, SCREEN_HEIGHT),
		D2D1::BitmapProperties(D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM,
			D2D1_ALPHA_MODE_IGNORE)), &memkeptarolo);
}

void CUDA_cleanup_main_buffer(void)
{
	cudaMemset(dev_image_data, 255, SCREEN_HEIGHT*SCREEN_WIDTH * sizeof(unsigned int));
}

__global__ void CUDA_CleanUp_Zbuffer(float *zpuffer)
{
	int i;
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (i = index; i < SCREEN_HEIGHT*SCREEN_WIDTH; i += stride)
	{
		zpuffer[i] = 999999;
	}
}

void swap_main_buffer(void)
{
	display_area.left = 0;
	display_area.top = 0;
	display_area.right = SCREEN_WIDTH;
	display_area.bottom = SCREEN_HEIGHT;
	memkeptarolo->CopyFromMemory(&display_area, image_data, SCREEN_WIDTH * sizeof(unsigned int));
	pRT->BeginDraw();
	pRT->DrawBitmap(memkeptarolo, D2D1::RectF(0.0f, 0.0f, SCREEN_WIDTH, SCREEN_HEIGHT), 1.0f, D2D1_BITMAP_INTERPOLATION_MODE_NEAREST_NEIGHBOR, NULL);
	pRT->EndDraw();
}

__device__ void CUDA_SetPixel(int x1, int y1, int color, unsigned int *puffer)
{
	puffer[(y1 * SCREEN_WIDTH) + x1] = color;
}

__device__ void CUDA_SetPixel_Zbuffer(int x1, int y1, int z1, int color, unsigned int *puffer, float *zpuffer)
{
	int offset = (y1 * SCREEN_WIDTH) + x1;
	if (zpuffer[offset] > z1)
	{
		zpuffer[offset] = z1;
		puffer[offset] = color;
	}
}

__device__ void CUDA_DrawLine(int x1, int y1, int x2, int y2, int color, unsigned int *puffer)
{
	bool flip = false;
	int swap, offset;

	if (abs(x2 - x1) < 2 && abs(y2 - y1) < 2)
	{
		puffer[(y2*SCREEN_WIDTH) + x2] = color; return;
	}
	if (abs(x1 - x2) < abs(y1 - y2))
	{
		swap = x1;
		x1 = y1;
		y1 = swap;

		swap = x2;
		x2 = y2;
		y2 = swap;
		flip = true;
	}
	if (x1 > x2)
	{
		swap = x1;
		x1 = x2;
		x2 = swap;

		swap = y1;
		y1 = y2;
		y2 = swap;
	}
	int dx = x2 - x1;
	int dy = y2 - y1;

	int marker1 = abs(dy) * 2;
	int marker2 = 0;
	int y = y1, x;

	if (flip)
	{
		for (x = x1; x <= x2; ++x)
		{
			offset = (x * SCREEN_WIDTH);
			puffer[offset + y] = color;
			marker2 += marker1;
			if (marker2 > dx)
			{
				y += (y2 > y1 ? 1 : -1);
				marker2 -= dx * 2;
			}
		}
	}
	else
	{
		for (x = x1; x <= x2; ++x)
		{
			offset = (y * SCREEN_WIDTH);
			puffer[offset + x] = color;
			marker2 += marker1;
			if (marker2 > dx)
			{
				y += (y2 > y1 ? 1 : -1);
				marker2 -= dx * 2;
			}
		}
	}
}

__device__ void CUDA_DrawLine_Zbuffer(int x1, int y1, int z1, int x2, int y2, int z2, int color, unsigned int *puffer, float *zpuffer)
{
	float Pz;
	bool flip = false;
	int swap, offset;

	if (abs(x2 - x1) < 2 && abs(y2 - y1) < 2) {
		puffer[(y2*SCREEN_WIDTH) + x2] = color; return;
	}
	if (abs(x1 - x2) < abs(y1 - y2))
	{
		swap = x1;
		x1 = y1;
		y1 = swap;

		swap = x2;
		x2 = y2;
		y2 = swap;
		flip = true;
	}
	if (x1 > x2)
	{
		swap = x1;
		x1 = x2;
		x2 = swap;

		swap = y1;
		y1 = y2;
		y2 = swap;
	}
	int dx = x2 - x1;
	int dy = y2 - y1;

	int marker1 = abs(dy) * 2;
	int marker2 = 0;
	int y = y1, x;

	for (x = x1; x <= x2; ++x)
	{
		if (z1 == z2) Pz = z1;
		else
		{
			int s1 = abs(x2 - x1);
			int s2 = abs(z1 - z2);
			Pz = (float)z2 + (float)((((float)x - (float)x1) / (float)s1) * (float)s2);
		}
		if (flip)
		{
			offset = (x * SCREEN_WIDTH);
			if (zpuffer[offset + y] > Pz)
			{
				zpuffer[offset + y] = Pz;
				puffer[offset + y] = color;
			}
		}
		else
		{
			offset = (y * SCREEN_WIDTH);
			if (zpuffer[offset + x] > Pz)
			{
				zpuffer[offset + x] = Pz;
				puffer[offset + x] = color;
			}
		}
		marker2 += marker1;
		if (marker2 > dx)
		{
			y += (y2 > y1 ? 1 : -1);
			marker2 -= dx * 2;
		}
	}
}

void CUDA_FillTriangle(int x1, int y1, int x2, int y2, int x3, int y3, int color, unsigned int *puffer)
{
	int Ax, Ay, Bx, By, i, j;
	int swapx, swapy, offset, maxoffset = SCREEN_HEIGHT * SCREEN_WIDTH;
	if (y1 == y2 && y1 == y3) return;

	if (y1 > y2)
	{
		swapx = x1;
		swapy = y1;
		x1 = x2;
		y1 = y2;
		x2 = swapx;
		y2 = swapy;
	}
	if (y1 > y3)
	{
		swapx = x1;
		swapy = y1;
		x1 = x3;
		y1 = y3;
		x3 = swapx;
		y3 = swapy;
	}
	if (y2 > y3)
	{
		swapx = x3;
		swapy = y3;
		x3 = x2;
		y3 = y2;
		x2 = swapx;
		y2 = swapy;
	}
	int t_height = y3 - y1;
	for (i = 0; i < t_height; ++i)
	{
		bool lower_part = i > y2 - y1 || y2 == y1;
		int part_height = lower_part ? y3 - y2 : y2 - y1;
		float alpha = (float)i / t_height;
		float beta = (float)(i - (lower_part ? y2 - y1 : 0)) / part_height;
		Ax = x1 + (x3 - x1)*alpha;
		Ay = y1 + (y3 - y1)*alpha;
		Bx = lower_part ? x2 + (x3 - x2)*beta : x1 + (x2 - x1)*beta;
		By = lower_part ? y2 + (y3 - y2)*beta : y1 + (y2 - y1)*beta;
		if (Ax > Bx)
		{
			swapx = Ax;
			swapy = Ay;
			Ax = Bx;
			Ay = By;
			Bx = swapx;
			By = swapy;
		}

		offset = (y1 + i)*SCREEN_WIDTH;
		for (j = Ax; j < Bx; ++j)
		{
			if (offset + j > maxoffset) continue;
			puffer[offset + j] = color;
		}
	}
}

__device__ void CUDA_FillTriangle_Zbuffer(int x1, int y1, int z1, int x2, int y2, int z2, int x3, int y3, int z3, int color, unsigned int *puffer, float *zpuffer)
{
	int Ax, Ay, Bx, By, i, j, depth_value;
	int swapx, swapy, offset;
	Vec3f interpolate, helper_vector;
	if (y1 == y2 && y1 == y3) return;

	if (y1 > y2)
	{
		swapx = x1;
		swapy = y1;
		x1 = x2;
		y1 = y2;
		x2 = swapx;
		y2 = swapy;
	}
	if (y1 > y3)
	{
		swapx = x1;
		swapy = y1;
		x1 = x3;
		y1 = y3;
		x3 = swapx;
		y3 = swapy;
	}
	if (y2 > y3)
	{
		swapx = x3;
		swapy = y3;
		x3 = x2;
		y3 = y2;
		x2 = swapx;
		y2 = swapy;
	}
	int t_height = y3 - y1;
	for (i = 0; i < t_height; ++i)
	{
		bool second_half = i > y2 - y1 || y2 == y1;
		int segment_height = second_half ? y3 - y2 : y2 - y1;
		float alpha = (float)i / t_height;
		float beta = (float)(i - (second_half ? y2 - y1 : 0)) / segment_height;
		Ax = x1 + (x3 - x1)*alpha;
		Ay = y1 + (y3 - y1)*alpha;
		Bx = second_half ? x2 + (x3 - x2)*beta : x1 + (x2 - x1)*beta;
		By = second_half ? y2 + (y3 - y2)*beta : y1 + (y2 - y1)*beta;
		if (Ax > Bx)
		{
			swapx = Ax;
			swapy = Ay;
			Ax = Bx;
			Ay = By;
			Bx = swapx;
			By = swapy;
		}

		offset = (y1 + i)*SCREEN_WIDTH;
		for (j = Ax; j <= Bx; ++j)
		{
			helper_vector.x = (x2 - x1) * (y1 - (y1 + i)) - (x1 - j) * (y2 - y1);
			helper_vector.y = (x1 - j) * (y3 - y1) - (x3 - x1) * (y1 - (y1 + i));
			helper_vector.z = (x3 - x1) * (y2 - y1) - (x2 - x1) * (y3 - y1);
			if (abs((int)helper_vector.z) < 1) { interpolate.x = -1; interpolate.y = 0; interpolate.z = 0; }
			else
			{
				interpolate.x = 1.f - (helper_vector.x + helper_vector.y) / helper_vector.z;
				interpolate.y = helper_vector.y / helper_vector.z;
				interpolate.z = helper_vector.x / helper_vector.z;
			}
			if (interpolate.x < 0 || interpolate.y < 0 || interpolate.z < 0) continue;
			depth_value = (z1*interpolate.x) + (z2*interpolate.y) + (z3*interpolate.z);
			if (zpuffer[offset + j] > depth_value)
			{
				zpuffer[offset + j] = depth_value;
				puffer[offset + j] = color;
			}
		}
	}
}

void init_3D(void)
{
	persp_degree = Math_PI / 180;
	rot_degree_x = 0 * Math_PI / 180; rot_degree_x2 = 0;
	rot_degree_y = 0 * Math_PI / 180; rot_degree_y2 = 0;
	rot_degree_z = 0 * Math_PI / 180; rot_degree_z2 = 0;
	cleanup_matrices();
}

void cleanup_matrices(void)
{
	raw_vertex_counter = 0;
	raw_vertices_length = 0;
}

void data_transfer_to_GPU(void)
{
	cudaMemcpy(dev_raw_verticesX, raw_verticesX, raw_vertices_length * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_raw_verticesY, raw_verticesY, raw_vertices_length * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_raw_verticesZ, raw_verticesZ, raw_vertices_length * sizeof(float), cudaMemcpyHostToDevice);
}

//********************************
//OBJ format handling
//********************************
int getelementcount(unsigned char csv_content[])
{
	int s1, s2;
	for (s1 = s2 = 0; s1 < strlen((const char *)csv_content); ++s1)
	{
		if (csv_content[s1] == 10) break;
		else if (csv_content[s1] == 32) ++s2;
	}
	return s2;
}

void getelement(unsigned char csv_content[], unsigned int data_index, unsigned char csv_content2[])
{
	int s1, s2, s3, s4 = 0;
	for (s1 = 0, s2 = 0; s1 < strlen((const char *)csv_content); ++s1)
	{
		if (csv_content[s1] == 32)
		{
			++s2;
			if (s2 == data_index)
			{
				for (s3 = s1 + 1; s3 < strlen((const char *)csv_content); ++s3)
				{
					if (csv_content[s3] == 32 || csv_content[s3] == 10)
					{
						csv_content2[s4] = 0;
						return;
					}
					else csv_content2[s4++] = csv_content[s3];
				}
			}
		}
	}
}

void obj_loader(void)
{
	FILE *objfile;
	int i, j;
	float data1, data2, data3;
	unsigned char row1[1024], row2[1024];
	int data_count, max_row_length = 250;
	char tempstr[200];

	objfile = fopen("ship.obj", "rt");
	if (objfile == NULL) return;

	vertex_counter = poly_counter = 0;
	tomb_vertices_length = tomb_vertices_length = 0;

	while (!feof(objfile))
	{
		fgets((char *)row1, max_row_length, objfile);

		if (row1[0] == 118 && row1[1] == 32) //*** 'v '
		{
			getelement(row1, 1, row2); data1 = atof((const char *)row2);
			getelement(row1, 2, row2); data2 = atof((const char *)row2);
			getelement(row1, 3, row2); data3 = atof((const char *)row2);
			tomb_vertices[tomb_vertices_length][0] = data1 * 4;
			tomb_vertices[tomb_vertices_length][1] = data2 * 4;
			tomb_vertices[tomb_vertices_length++][2] = data3 * 4;
		}
		else if (row1[0] == 102 && row1[1] == 32) //*** 'f '
		{
			data_count = getelementcount(row1);

			tomb_faces[tomb_faces_length][0] = data_count;
			for (i = 1; i < data_count + 1; ++i)
			{
				getelement(row1, i, row2);
				data1 = atof((const char *)row2);
				tomb_faces[tomb_faces_length][i] = data1 - 1;
			}
			++tomb_faces_length;
		}
	}
	fclose(objfile);
	int  base_index;
	for (i = 0; i < tomb_faces_length; ++i)
	{
		base_index = tomb_faces[i][1];
		if (tomb_faces[i][0] == 3)
		{
			raw_verticesX[raw_vertices_length] = tomb_vertices[tomb_faces[i][1]][0];
			raw_verticesY[raw_vertices_length] = tomb_vertices[tomb_faces[i][1]][1];
			raw_verticesZ[raw_vertices_length++] = tomb_vertices[tomb_faces[i][1]][2];

			raw_verticesX[raw_vertices_length] = tomb_vertices[tomb_faces[i][2]][0];
			raw_verticesY[raw_vertices_length] = tomb_vertices[tomb_faces[i][2]][1];
			raw_verticesZ[raw_vertices_length++] = tomb_vertices[tomb_faces[i][2]][2];

			raw_verticesX[raw_vertices_length] = tomb_vertices[tomb_faces[i][3]][0];
			raw_verticesY[raw_vertices_length] = tomb_vertices[tomb_faces[i][3]][1];
			raw_verticesZ[raw_vertices_length++] = tomb_vertices[tomb_faces[i][3]][2];
			++poly_counter;
			vertex_counter += 3;
		}
		else if (tomb_faces[i][0] == 4)
		{
			raw_verticesX[raw_vertices_length] = tomb_vertices[tomb_faces[i][1]][0];
			raw_verticesY[raw_vertices_length] = tomb_vertices[tomb_faces[i][1]][1];
			raw_verticesZ[raw_vertices_length++] = tomb_vertices[tomb_faces[i][1]][2];

			raw_verticesX[raw_vertices_length] = tomb_vertices[tomb_faces[i][2]][0];
			raw_verticesY[raw_vertices_length] = tomb_vertices[tomb_faces[i][2]][1];
			raw_verticesZ[raw_vertices_length++] = tomb_vertices[tomb_faces[i][2]][2];

			raw_verticesX[raw_vertices_length] = tomb_vertices[tomb_faces[i][3]][0];
			raw_verticesY[raw_vertices_length] = tomb_vertices[tomb_faces[i][3]][1];
			raw_verticesZ[raw_vertices_length++] = tomb_vertices[tomb_faces[i][3]][2];

			raw_verticesX[raw_vertices_length] = tomb_vertices[tomb_faces[i][1]][0];
			raw_verticesY[raw_vertices_length] = tomb_vertices[tomb_faces[i][1]][1];
			raw_verticesZ[raw_vertices_length++] = tomb_vertices[tomb_faces[i][1]][2];

			raw_verticesX[raw_vertices_length] = tomb_vertices[tomb_faces[i][3]][0];
			raw_verticesY[raw_vertices_length] = tomb_vertices[tomb_faces[i][3]][1];
			raw_verticesZ[raw_vertices_length++] = tomb_vertices[tomb_faces[i][3]][2];

			raw_verticesX[raw_vertices_length] = tomb_vertices[tomb_faces[i][4]][0];
			raw_verticesY[raw_vertices_length] = tomb_vertices[tomb_faces[i][4]][1];
			raw_verticesZ[raw_vertices_length++] = tomb_vertices[tomb_faces[i][4]][2];
			poly_counter += 2;
			vertex_counter += 6;
		}
	}
}

void D2D_drawing(void)
{
	if (drawing_in_progress == 1) return;
	drawing_in_progress = 1;
	char tempstr[255], tempstr2[255], hibauzenet[256];
	int blockSize = 384;
	int numBlocks = (raw_vertices_length + blockSize - 1) / blockSize;

	strcpy(tempstr2, "Vertices: ");
	_itoa(vertex_counter, tempstr, 10); strcat(tempstr2, tempstr); strcat(tempstr2, " Polygons: ");
	_itoa(poly_counter, tempstr, 10); strcat(tempstr2, tempstr); strcat(tempstr2, " Z ordered: ");

	starttime = GetTickCount();

	rot_degree_x = rot_degree_x2 * Math_PI / 180;
	rot_degree_y = rot_degree_y2 * Math_PI / 180;
	rot_degree_z = rot_degree_z2 * Math_PI / 180;
	float degree_sinx = sin(rot_degree_x);
	float degree_cosx = cos(rot_degree_x);
	float degree_siny = sin(rot_degree_y);
	float degree_cosy = cos(rot_degree_y);
	float degree_sinz = sin(rot_degree_z);
	float degree_cosz = cos(rot_degree_z);
	CUDA_rotation << <numBlocks, blockSize >> > (raw_vertices_length, dev_raw_verticesX, dev_raw_verticesY, dev_raw_verticesZ, dev_rotated_verticesX, dev_rotated_verticesY, dev_rotated_verticesZ, degree_cosx, degree_sinx, degree_cosy, degree_siny, degree_cosz, degree_sinz);
	cudaDeviceSynchronize();

	strcpy_s(hibauzenet, cudaGetErrorString(cudaGetLastError()));
	drawing();

	endtime = GetTickCount();
	if ((endtime - starttime) == 0) ++endtime;
	fps_stat = 1000 / (endtime - starttime); strcat(tempstr2, " FPS: "); _itoa(fps_stat, tempstr, 10); strcat(tempstr2, tempstr);
	strcat(tempstr2, ", X: "); _itoa(rot_degree_x2, tempstr, 10); strcat(tempstr2, tempstr);
	strcat(tempstr2, ", Y: "); _itoa(rot_degree_y2, tempstr, 10); strcat(tempstr2, tempstr);
	strcat(tempstr2, ", Z: "); _itoa(rot_degree_z2, tempstr, 10); strcat(tempstr2, tempstr);
	strcat(tempstr2, ", CUDA: "); strcat(tempstr2, hibauzenet);
	SetWindowTextA(Form1, tempstr2);
	drawing_in_progress = 0;
}

__global__ void CUDA_rotation(int maxitemcount, float *rawarrayX, float *rawarrayY, float *rawarrayZ, float *rotarrayX, float *rotarrayY, float *rotarrayZ, float degree_cosx, float degree_sinx, float degree_cosy, float degree_siny, float degree_cosz, float degree_sinz)
{
	int i;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	float t0;

	//rotaion
	for (i = index; i < maxitemcount; i += stride)
	{
		rotarrayY[i] = (rawarrayY[i] * degree_cosx) - (rawarrayZ[i] * degree_sinx);
		rotarrayZ[i] = rawarrayY[i] * degree_sinx + rawarrayZ[i] * degree_cosx;

		rotarrayX[i] = rawarrayX[i] * degree_cosy + rotarrayZ[i] * degree_siny;
		rotarrayZ[i] = -rawarrayX[i] * degree_siny + rotarrayZ[i] * degree_cosy;// +

		t0 = rotarrayX[i];
		//some tweaking for OBJ models: "+ (SCREEN_WIDTH / 4)" and "+ (SCREEN_HEIGHT / 4)"
		rotarrayX[i] = t0 * degree_cosz - rotarrayY[i] * degree_sinz  + (SCREEN_WIDTH / 4);
		rotarrayY[i] = t0 * degree_sinz + rotarrayY[i] * degree_cosz  + (SCREEN_HEIGHT / 4);
	}

	//perspective projection
	int s1;
	int viewpoint = -1100;
	float sx = SCREEN_WIDTH / 2;
	float sultra = SCREEN_HEIGHT / 2, sultra2 = SCREEN_HEIGHT / 3;
	int x_minusz_edge = 0, y_minusz_edge = 0, x_max_edge = SCREEN_WIDTH - 1, y_max_edge = SCREEN_HEIGHT - 1;
	float distance;

	for (i = index; i < maxitemcount; i += stride)
	{
		distance = 999999;

		if (rotarrayZ[i] < distance) distance = rotarrayZ[i];
		if (distance < viewpoint) { rotarrayZ[i] = -9999999; continue; }
		sultra = viewpoint / (viewpoint - rotarrayZ[i]);
		rotarrayX[i] = rotarrayX[i] * sultra + 400;
		rotarrayY[i] = (rotarrayY[i] * sultra) + sultra2;
		if (rotarrayX[i] < x_minusz_edge || rotarrayX[i] > x_max_edge) { rotarrayZ[i] = -9999999; continue; }
		if (rotarrayY[i] < y_minusz_edge || rotarrayY[i] > y_max_edge) { rotarrayZ[i] = -9999999; continue; }
	}
}

void drawing(void)
{
	CUDA_cleanup_main_buffer();
	CUDA_CleanUp_Zbuffer << < ((SCREEN_WIDTH*SCREEN_HEIGHT) + 384 - 1) / 384, 384 >> > (dev_zbuffer);
	cudaDeviceSynchronize();
	render_objects << <12, 384 >> > (raw_vertices_length, dev_rotated_verticesX, dev_rotated_verticesY, dev_rotated_verticesZ, dev_image_data, dev_zbuffer);
	cudaDeviceSynchronize();

	cudaMemcpy(image_data, dev_image_data, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	swap_main_buffer();
}

__global__ void render_objects(int maxitemcount, float *rotarrayX, float *rotarrayY, float *rotarrayZ, unsigned int *puffer, float *zpuffer)
{
	int i, px, py, drawcolor;
	int index = (blockIdx.x * blockDim.x) + (threadIdx.x * 3);
	int stride = blockDim.x * gridDim.x;
	//VEKTOR Vector1, Vector2, vNormal;//for visibility check

	for (i = index; i < maxitemcount - 3; i += stride)
	{
		if ((rotarrayZ[i] < -9000000) || (rotarrayZ[i + 1] < -9000000) || (rotarrayZ[i + 2] < -9000000)) continue;

		/* for visibility check
		Vector1.x = rotarrayX[i + 1] - rotarrayX[i];
		Vector1.y = rotarrayY[i + 1] - rotarrayY[i];
		Vector1.z = rotarrayZ[i + 1] - rotarrayZ[i];
		Vector2.x = rotarrayX[i + 2] - rotarrayX[i];
		Vector2.y = rotarrayY[i + 2] - rotarrayY[i];
		Vector2.z = rotarrayZ[i + 2] - rotarrayZ[i];

		vNormal.x = ((Vector1.y * Vector2.z) - (Vector1.z * Vector2.y));
		vNormal.y = ((Vector1.z * Vector2.x) - (Vector1.x * Vector2.z));
		vNormal.z = ((Vector1.x * Vector2.y) - (Vector1.y * Vector2.x));
		if (vNormal.z > 0) continue;
		*/

		drawcolor = RGB(180 * ((float)i / (float)maxitemcount * 100), 180 * ((float)i / (float)maxitemcount * 100), 180 * ((float)i / (float)maxitemcount * 100));
		//CUDA_SetPixel(rotarrayX[i], rotarrayY[i], RGB(0, 0, 0),puffer);
		//CUDA_SetPixel_Zbuffer(rotarrayX[i], rotarrayY[i], rotarrayZ[i], drawcolor, puffer, zpuffer);

		/*CUDA_DrawLine(rotarrayX[i], rotarrayY[i], rotarrayX[i + 1], rotarrayY[i + 1], RGB(0, 0, 0), puffer);
		CUDA_DrawLine(rotarrayX[i+2], rotarrayY[i+2], rotarrayX[i + 1], rotarrayY[i + 1], RGB(0, 0, 0), puffer);
		CUDA_DrawLine(rotarrayX[i], rotarrayY[i], rotarrayX[i + 2], rotarrayY[i + 2], RGB(0, 0, 0), puffer);//*/

		/*CUDA_DrawLine_Zbuffer(rotarrayX[i], rotarrayY[i], rotarrayZ[i], rotarrayX[i + 1], rotarrayY[i + 1], rotarrayZ[i+1], RGB(0, 0, 0), puffer,zpuffer);
		CUDA_DrawLine_Zbuffer(rotarrayX[i + 2], rotarrayY[i + 2], rotarrayZ[i+2], rotarrayX[i + 1], rotarrayY[i + 1], rotarrayZ[i+1], RGB(0, 0, 0), puffer, zpuffer);
		CUDA_DrawLine_Zbuffer(rotarrayX[i], rotarrayY[i], rotarrayZ[i], rotarrayX[i + 2], rotarrayY[i + 2], rotarrayZ[i+2], RGB(0, 0, 0), puffer, zpuffer);//*/

		//CUDA_FillTriangle(rotarrayX[i], rotarrayY[i], rotarrayX[i + 1], rotarrayY[i + 1], rotarrayX[i + 2], rotarrayY[i + 2], RGB(i*0.05, i*0.05, i*0.05), puffer);
		CUDA_FillTriangle_Zbuffer(rotarrayX[i], rotarrayY[i], rotarrayZ[i], rotarrayX[i + 1], rotarrayY[i + 1], rotarrayZ[i + 1], rotarrayX[i + 2], rotarrayY[i + 2], rotarrayZ[i + 2], drawcolor, puffer, zpuffer);

	}
}

__global__ void zoom_in(int maxitemcount, float *rawarrayX, float *rawarrayY, float *rawarrayZ)
{
	int i;
	int index = (blockIdx.x * blockDim.x) + (threadIdx.x * 1);
	int stride = blockDim.x * gridDim.x;
	for (i = index; i < maxitemcount; i += stride)
	{
		rawarrayX[i] *= 1.2;
		rawarrayY[i] *= 1.2;
		rawarrayZ[i] *= 1.2;
	}
}

__global__ void zoom_out(int maxitemcount, float *rawarrayX, float *rawarrayY, float *rawarrayZ)
{
	int i;
	int index = (blockIdx.x * blockDim.x) + (threadIdx.x * 1);
	int stride = blockDim.x * gridDim.x;
	for (i = index; i < maxitemcount; i += stride)
	{
		rawarrayX[i] /= 1.2;
		rawarrayY[i] /= 1.2;
		rawarrayZ[i] /= 1.2;
	}
}
