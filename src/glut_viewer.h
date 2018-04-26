/*
 * Fermat
 *
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <renderer.h>
#include <GL/freeglut.h>
#include <buffers.h>
#include <vector>
#include <cugar/basic/threads.h>

// A class encpasulating our GLUT renderer
//
struct GlutViewer : public Renderer
{
	enum Orientation { X_UP, Y_UP, Z_UP };

	void init(int argc, char** argv);

	static void display();
	static void idle();
	static void mouse(int button, int state, int x, int y);
	static void motion(int x, int y);
	static void keyboard(unsigned char character, int x, int y);

	GLuint					m_texture;
	Camera					m_camera_o;
	DiskLight				m_light_o;
	int2					m_mouse;
	bool					m_dollying;
	bool					m_panning;
	bool					m_walking;
	bool					m_zooming;
	bool					m_moving_selection;
	bool					m_light_zooming;
	Orientation				m_orientation;
	bool					m_dirty;
	bool					m_dirty_geo;
	uint32					m_instance;
	uint32					m_selected_group;
	char					m_output_name[1024];
	FILE*					m_ffmpeg;
	FILE*					m_camera_path;
	bool					m_record;
	bool					m_playback;

	cugar::Mutex mutex;
};

// global Renderer instance
extern GlutViewer* s_renderer;
