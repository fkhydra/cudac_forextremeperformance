#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include <d2d1.h>
#include <d2d1helper.h>
#pragma comment(lib, "d2d1")

//*****double buffering*****
#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1000

D2D1_RECT_U display_area;
ID2D1Bitmap *image_container = NULL;
unsigned int *dev_image_data, image_data[SCREEN_WIDTH * SCREEN_HEIGHT];
float *dev_zbuffer;
typedef struct Vec3f {
	float x, y, z;
};
//**************************************

ID2D1Factory* pD2DFactory = NULL;
ID2D1HwndRenderTarget* pRT = NULL;

#define HIBA_00 TEXT("Error:Program initialisation process.")
HINSTANCE hInstGlob;
int SajatiCmdShow;
char szClassName[] = "WindowsApp";
HWND Form1; //Windows handler

LRESULT CALLBACK WndProc0(HWND, UINT, WPARAM, LPARAM);
void D2D_drawing(ID2D1HwndRenderTarget* pRT);

//*****double buffering*****
void create_main_buffer(void);
void CUDA_cleanup_main_buffer(void);
__global__ void CUDA_CleanUp_Zbuffer(float *zpuffer);
void swap_main_buffer(void);
//**************************************

//*****Drawing algorithms*****
__device__ void CUDA_SetPixel_Zbuffer(int x1, int y1, int z1, int color, unsigned int *puffer, float *zpuffer);
__device__ void CUDA_DrawLine_Zbuffer(int x1, int y1, int z1, int x2, int y2, int z2, int color, unsigned int *puffer, float *zpuffer);
__device__ void CUDA_FillTriangle_Zbuffer(int x1, int y1, int z1, int x2, int y2, int z2, int x3, int y3, int z3, int color, unsigned int *puffer, float *zpuffer);
//**************************************

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
		cudaMalloc((void**)&dev_image_data, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(unsigned int));
		cudaMalloc((void**)&dev_zbuffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(float));
		return 0;
	//*********************************
	//to eliminate color flickering
	//*********************************
	case WM_ERASEBKGND:
		return (LRESULT)1;
	//*********************************
	//Repainting the client area of the window
	//*********************************
	case WM_PAINT:
		hdc = BeginPaint(hwnd, &ps);
		EndPaint(hwnd, &ps);
		D2D_drawing(pRT);
		return 0;
	//*********************************
	//Closing the window, freeing resources
	//*********************************
	case WM_CLOSE:
		pRT->Release();
		pD2DFactory->Release();
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

void D2D_drawing(ID2D1HwndRenderTarget* pRT)
{
	CUDA_cleanup_main_buffer();
	CUDA_CleanUp_Zbuffer <<< ((SCREEN_WIDTH*SCREEN_HEIGHT) + 384 - 1) / 384, 384 >>> (dev_zbuffer);
	cudaDeviceSynchronize();
	//render_objects<<<blocks,threads >>>(dev_image_data,dev_zbuffer);
	cudaDeviceSynchronize();
	cudaMemcpy(image_data, dev_image_data, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	swap_main_buffer();
}

void create_main_buffer(void)
{
	pRT->CreateBitmap(D2D1::SizeU(SCREEN_WIDTH, SCREEN_HEIGHT),
		D2D1::BitmapProperties(D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM,
			D2D1_ALPHA_MODE_IGNORE)), &image_container);
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
	image_container->CopyFromMemory(&display_area, image_data, SCREEN_WIDTH * sizeof(unsigned int));
	pRT->BeginDraw();
	pRT->DrawBitmap(image_container, D2D1::RectF(0.0f, 0.0f, SCREEN_WIDTH, SCREEN_HEIGHT), 1.0f, D2D1_BITMAP_INTERPOLATION_MODE_NEAREST_NEIGHBOR, NULL);
	pRT->EndDraw();
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