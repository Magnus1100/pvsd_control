/**
 * \file   defines.h
 *
 * \brief  Define statements used by the FMU methods.
 *
 * \author Thierry S. Nouidui,
 *         Simulation Research Group,
 *         LBNL,
 *         TSNouidui@lbl.gov
 *
 * \date   2012-09-03
 *
 */

#ifndef __DEFINES_H__
#define __DEFINES_H__

/** \val Debug flag, uncomment to enable debug messages

#ifndef NDEBUG
#define NDEBUG
#endif
*/

/** \val Lenght of buffer used to communicate with the BSD socket. */
#define BUFFER_LENGTH  1024

/** \val The main version of the socket interface.
 */
#define MAINVERSION 2


/////////////////////////////////////////////////////////////////////
/*  Header specific to the FMU export project (added by T. Nouidui)
*   Filename for input file and weather file
*/
/////////////////////////////////////////////////////////////////////
#define FRUNINFILE   "runinfile.idf"
#define FRUNWEAFILE  "runweafile.epw"
#define FTIMESTEP    "tstep.txt"
#define VARCFG       "variables.cfg"
#define SOCKCFG      "socket.cfg"
#define EPBAT        "EP.bat"
#define MAX_VARNAME_LEN 100
#ifdef _MSC_VER
#include <windows.h>
#define PATH_SEP "\\"
#else
#define PATH_SEP "//"
#endif
#define RESOURCES "resources"
#define XML_FILE "modelDescription.xml"

#include "../fmusdk-shared/include/fmi2TypesPlatform.h"
#include "../fmusdk-shared/include/fmi2Functions.h"
#include "../fmusdk-shared/parser/XmlParserCApi.h"


/* Export fmi functions on Windows */
#ifdef _MSC_VER
#define DllExport __declspec( dllexport )
#else
#define DllExport
#endif

typedef struct ModelInstance {
	int index;
	const fmi2CallbackFunctions* functions;
	fmi2ComponentEnvironment componentEnvironment;
	char instanceName[MAX_VARNAME_LEN];
	char cwd[256];
	char in_file_name[100];
	char* wea_file;
	char* in_file;
	char* idd_file;
	char* fmuResourceLocation;
	char* fmuUnzipLocation;
	char *xml_file;
	char* tmpResCon;
	char *fmuOutput;
	char* mID;
	char *mGUID;
	int numInVar;
	int numOutVar;
	int sockfd;
	int newsockfd;

	fmi2Boolean visible;
	fmi2Boolean loggingOn;


	int firstCallGetReal;
	int firstCallSetReal;
	int firstCallDoStep;

	int firstCallFree;
	int firstCallTerm;
	int firstCallIni;
	int firstCallRes;
	int flaGetRealCall;
	int flaGetWri;
	int flaGetRea;
	int flaWri;
	int flaRea;
	int readReady;
	int writeReady;
	int timeStepIDF;
	int getCounter;
	int setCounter;
	int setupExperiment;

	ModelDescription * md;
	fmi2Real *inVec;
	fmi2Real *outVec;
	fmi2Real tStartFMU;
	fmi2Real tStopFMU;
	fmi2Real nexComm;
	fmi2Real simTimSen;
	fmi2Real simTimRec;
	fmi2Real communicationStepSize;
	fmi2Real curComm;

#ifdef _MSC_VER
	HANDLE  pid;
#else
	pid_t  pid;
#endif
} ModelInstance;

#endif /*__DEFINES_H__*/


/*

***********************************************************************************
Copyright Notice
----------------

Functional Mock-up Unit Export of EnergyPlus �2013, The Regents of
the University of California, through Lawrence Berkeley National
Laboratory (subject to receipt of any required approvals from
the U.S. Department of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Technology Transfer Department at
TTD@lbl.gov.referring to "Functional Mock-up Unit Export
of EnergyPlus (LBNL Ref 2013-088)".

NOTICE: This software was produced by The Regents of the
University of California under Contract No. DE-AC02-05CH11231
with the Department of Energy.
For 5 years from November 1, 2012, the Government is granted for itself
and others acting on its behalf a nonexclusive, paid-up, irrevocable
worldwide license in this data to reproduce, prepare derivative works,
and perform publicly and display publicly, by or on behalf of the Government.
There is provision for the possible extension of the term of this license.
Subsequent to that period or any extension granted, the Government is granted
for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable
worldwide license in this data to reproduce, prepare derivative works,
distribute copies to the public, perform publicly and display publicly,
and to permit others to do so. The specific term of the license can be identified
by inquiry made to Lawrence Berkeley National Laboratory or DOE. Neither
the United States nor the United States Department of Energy, nor any of their employees,
makes any warranty, express or implied, or assumes any legal liability or responsibility
for the accuracy, completeness, or usefulness of any data, apparatus, product,
or process disclosed, or represents that its use would not infringe privately owned rights.


Copyright (c) 2013, The Regents of the University of California, Department
of Energy contract-operators of the Lawrence Berkeley National Laboratory.
All rights reserved.

1. Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the copyright notice, this list
of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the copyright notice, this list
of conditions and the following disclaimer in the documentation and/or other
materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

2. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

3. You are under no obligation whatsoever to provide any bug fixes, patches,
or upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National Laboratory,
without imposing a separate written license agreement for such Enhancements,
then you hereby grant the following license: a non-exclusive, royalty-free
perpetual license to install, use, modify, prepare derivative works, incorporate
into other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.

NOTE: This license corresponds to the "revised BSD" or "3-clause BSD"
License and includes the following modification: Paragraph 3. has been added.


***********************************************************************************
*/
