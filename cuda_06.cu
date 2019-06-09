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
unsigned int image_data[SCREEN_WIDTH * SCREEN_HEIGHT];
//**************************************

ID2D1Factory* pD2DFactory = NULL;
ID2D1HwndRenderTarget* pRT = NULL;

#define HIBA_00 TEXT("Error:Program initialisation process.")
HINSTANCE hInstGlob;
int SajatiCmdShow;
char szClassName[] = "WindowsApp";
HWND Form1; //Windows handler

LRESULT CALLBACK WndProc0(HWND, UINT, WPARAM, LPARAM);
void D2D_rajzolas(ID2D1HwndRenderTarget* pRT);

//*****double bufferingz*****
void create_main_buffer(void);
void cleanup_main_buffer(void);
void swap_main_buffer(void);
//**************************************

//*****Drawing algorithms*****
void SetPixel_main_buffer(int x, int y, unsigned int szin);
void DrawLine_main_buffer(int x1, int y1, int x2, int y2, int szin);
void FillTriangle_main_buffer(int x1, int y1, int x2, int y2, int x3, int y3, int szin);
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
	//Creating the windows
	//*********************************
	Form1 = CreateWindow(TEXT("WIN0"),
		TEXT("CUDA - DIRECT2D"),
		(WS_OVERLAPPED | WS_SYSMENU | WS_THICKFRAME | WS_MAXIMIZEBOX | WS_MINIMIZEBOX),
		50,
		50,
		400,
		300,
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
				hwnd, D2D1::SizeU(800, 600)),
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
		D2D_rajzolas(pRT);
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
	SetPixel_main_buffer(100, 100, RGB(0, 0, 0));
	DrawLine_main_buffer(10, 10, 300, 80, RGB(0, 0, 0));
	FillTriangle_main_buffer(500, 30, 600, 80, 530, 200, RGB(0, 0, 0));
	swap_main_buffer();
}

void create_main_buffer(void)
{
	pRT->CreateBitmap(D2D1::SizeU(SCREEN_WIDTH, SCREEN_HEIGHT),
		D2D1::BitmapProperties(D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM,
			D2D1_ALPHA_MODE_IGNORE)), &image_container);
}

void cleanup_main_buffer(void)
{
	memset(image_data, 255, SCREEN_HEIGHT*SCREEN_WIDTH * sizeof(unsigned int));
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

void SetPixel_main_buffer(int x, int y, unsigned int color)
{
	image_data[(y * SCREEN_WIDTH) + x] = color;
}

void DrawLine_main_buffer(int x1, int y1, int x2, int y2, int color)
{
	int swap, offset;
	bool flip = false;
	if (y2 < 0 || y1 < 0) return;

	if (abs(x2 - x1) < 2 && abs(y2 - y1) < 2) {
		image_data[(y2*SCREEN_WIDTH) + x2] = color; return;
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
			image_data[offset + y] = color;
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
			image_data[offset + x] = color;
			marker2 += marker1;
			if (marker2 > dx)
			{
				y += (y2 > y1 ? 1 : -1);
				marker2 -= dx * 2;
			}
		}
	}
}

void FillTriangle_main_buffer(int x1, int y1, int x2, int y2, int x3, int y3, int color)
{
	int Ax, Ay, Bx, By, j;
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
	int magassag = y3 - y1;

	for (int i = 0; i < magassag; ++i)
	{
		bool alsoresz = i > y2 - y1 || y2 == y1;
		int reszmagassag = alsoresz ? y3 - y2 : y2 - y1;
		float alpha = (float)i / magassag;
		float beta = (float)(i - (alsoresz ? y2 - y1 : 0)) / reszmagassag;
		Ax = x1 + (x3 - x1)*alpha;
		Ay = y1 + (y3 - y1)*alpha;
		Bx = alsoresz ? x2 + (x3 - x2)*beta : x1 + (x2 - x1)*beta;
		By = alsoresz ? y2 + (y3 - y2)*beta : y1 + (y2 - y1)*beta;
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
			image_data[offset + j] = color;
		}
	}
}