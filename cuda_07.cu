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
ID2D1Bitmap *memkeptarolo = NULL;
unsigned int image_data[SCREEN_WIDTH * SCREEN_HEIGHT];
float zbuffer[SCREEN_WIDTH*SCREEN_HEIGHT];
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
void cleanup_main_buffer(void);
void CleanUp_Zbuffer(void);
void swap_main_buffer(void);
//**************************************

//*****Drawing algorithms*****
void SetPixel_Zbuffer(int x1, int y1, int z1, int color);
void DrawLine_Zbuffer(int x0, int y0, int z0, int x1, int y1, int z1, int color);
void FillTriangle_Zbuffer(int x1, int y1, int z1, int x2, int y2, int z2, int x3, int y3, int z3, int color);
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
	cleanup_main_buffer();
	CleanUp_Zbuffer();
	SetPixel_Zbuffer(100,100,0,RGB(0,0,0));
	FillTriangle_Zbuffer(0, 0, 10, 600, 80, 20, 50, 400, 20, RGB(200, 200, 200));
	FillTriangle_Zbuffer(100, 30, 1, 200, 80, 1, 50, 90, 1, RGB(250, 0, 0));
	DrawLine_Zbuffer(10,10,10,300,80,10, RGB(0, 0, 0));	
	swap_main_buffer();
}

void create_main_buffer(void)
{
	pRT->CreateBitmap(D2D1::SizeU(SCREEN_WIDTH, SCREEN_HEIGHT),
		D2D1::BitmapProperties(D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM,
			D2D1_ALPHA_MODE_IGNORE)), &memkeptarolo);
}

void cleanup_main_buffer(void)
{
	memset(image_data, 255, SCREEN_HEIGHT*SCREEN_WIDTH * sizeof(unsigned int));
}

void CleanUp_Zbuffer(void)
{
	int i, j;
	for (i = 0; i < SCREEN_WIDTH; ++i)
		for (j = 0; j < SCREEN_HEIGHT; ++j)
		{
			zbuffer[(j * SCREEN_WIDTH) + i] = 9999999;
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

void SetPixel_Zbuffer(int x1, int y1, int z1, int color)
{
	int offset = (y1 * SCREEN_WIDTH) + x1;

	if (zbuffer[offset] > z1)
	{
		zbuffer[offset] = z1;
		image_data[offset] = color;
	}
}

void DrawLine_Zbuffer(int x0, int y0, int z0, int x1, int y1, int z1, int color)
{
	bool flip = false;
	int swap, offset;
	float depth_value;
	if (y1 < 0 || y0 < 0) return;

	if (abs(x0 - x1) < abs(y0 - y1))
	{
		swap = x0;
		x0 = y0;
		y0 = swap;

		swap = x1;
		x1 = y1;
		y1 = swap;
		flip = true;
	}
	if (x0 > x1)
	{
		swap = x0;
		x0 = x1;
		x1 = swap;

		swap = y0;
		y0 = y1;
		y1 = swap;
	}
	int dx = x1 - x0;
	int dy = y1 - y0;

	int marker1 = abs(dy) * 2;
	int marker2 = 0;
	int y = y0, x;

	for (x = x0; x <= x1; ++x)
	{
		if (z0 == z1) depth_value = z0;
		else
		{
			int s1 = abs(x1 - x0);
			int s2 = abs(z0 - z1);
			depth_value = (float)z1 + (float)((((float)x - (float)x0) / (float)s1) * (float)s2);
		}
		if (flip)
		{
			offset = (x * SCREEN_WIDTH);
			if (zbuffer[offset + y] > depth_value)
			{
				zbuffer[offset + y] = depth_value;
				image_data[offset + y] = color;
			}
		}
		else
		{
			offset = (y * SCREEN_WIDTH);
			if (zbuffer[offset + x] > depth_value)
			{
				zbuffer[offset + x] = depth_value;
				image_data[offset + x] = color;
			}
		}
		marker2 += marker1;
		if (marker2 > dx)
		{
			y += (y1 > y0 ? 1 : -1);
			marker2 -= dx * 2;
		}
	}
}

void FillTriangle_Zbuffer(int x1, int y1, int z1, int x2, int y2, int z2, int x3, int y3, int z3, int color)
{
	int Px, Py, depth_value, boxminx = SCREEN_WIDTH - 1, boxminy = SCREEN_HEIGHT - 1, boxmaxx = 0, boxmaxy = 0;
	int offset;
	Vec3f interpolate, helper_vector;

	if (y1 == y2 && y1 == y3) return;
	if (x1 == x2 && x1 == x3) return;

	boxminx = __min(x1, x2); boxminx = __min(boxminx, x3);
	boxminy = __min(y1, y2); boxminy = __min(boxminy, y3);
	boxmaxx = __max(x1, x2); boxmaxx = __max(boxmaxx, x3);
	boxmaxy = __max(y1, y2); boxmaxy = __max(boxmaxy, y3);

	for (Px = boxminx; Px <= boxmaxx; ++Px)
	{
		for (Py = boxminy; Py <= boxmaxy; ++Py)
		{
			offset = Px + (Py * SCREEN_WIDTH);
			helper_vector.x = (x2 - x1) * (y1 - Py) - (x1 - Px) * (y2 - y1);
			helper_vector.y = (x1 - Px) * (y3 - y1) - (x3 - x1) * (y1 - Py);
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
			if (zbuffer[offset] > depth_value)
			{
				zbuffer[offset] = depth_value;
				image_data[offset] = color;
			}
		}
	}
}