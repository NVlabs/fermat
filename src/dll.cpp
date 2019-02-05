//
//Copyright (c) 2016 NVIDIA Corporation.  All rights reserved.
//
//NVIDIA Corporation and its licensors retain all intellectual property and
//proprietary rights in and to this software, related documentation and any
//modifications thereto.  Any use, reproduction, disclosure or distribution of
//this software and related documentation without an express license agreement
//from NVIDIA Corporation is strictly prohibited.
//
//TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
//*AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
//OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
//MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
//NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
//CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
//LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
//INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
//POSSIBILITY OF SUCH DAMAGES
//

#pragma once

#include <dll.h>
#include <windows.h>
#include <cugar/basic/exceptions.h>

struct DLLImpl
{
	DLLImpl() {}

	DLLImpl(const char* filename)
	{
		size_t wsize = strlen(filename) + 1;

		wchar_t * wcstring = new wchar_t[wsize];

		// Convert char* string to a wchar_t* string.
		size_t convertedChars = 0;
		mbstowcs_s(&convertedChars, wcstring, wsize, filename, _TRUNCATE);
	
		hGetProcIDDLL = LoadLibrary(wcstring);
		if (!hGetProcIDDLL)
		{
			fprintf(stderr, "failed loading DLL: %s\n", filename);
			throw cugar::runtime_error("failed loading DLL");
		}
	}

	~DLLImpl()
	{
		FreeLibrary(hGetProcIDDLL); 
	}

	void* get_proc_address(const char* name)
	{
		FARPROC r = GetProcAddress(hGetProcIDDLL, name);
		if (r == NULL)
			fprintf(stderr, "failed loading %s\n", name);
		return (void*)r;
	}

	HINSTANCE hGetProcIDDLL;
};


DLL::DLL() :
	m_impl( new DLLImpl() )
{}
DLL::DLL(const char* filename) :
	m_impl( new DLLImpl(filename) )
{
}

DLL::~DLL() {}

void* DLL::get_proc_address(const char* name)
{
	return m_impl->get_proc_address( name );
}
