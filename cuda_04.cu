#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include <d2d1.h>
#include <d2d1helper.h>
#pragma comment(lib, "d2d1")

ID2D1Factory* pD2DFactory = NULL;
ID2D1HwndRenderTarget* pRT = NULL;

#define HIBA_00 TEXT("Error:Program initialisation process.")
HINSTANCE hInstGlob;
int SajatiCmdShow;
char szClassName[] = "WindowsApp";
HWND Form1; //Windows handle

LRESULT CALLBACK WndProc0(HWND, UINT, WPARAM, LPARAM);
void D2D_drawing(ID2D1HwndRenderTarget* pRT);

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
	pRT->BeginDraw();
	//
	pRT->EndDraw();
}
