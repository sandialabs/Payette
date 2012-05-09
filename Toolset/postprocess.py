#!/usr/bin/env python
# The MIT License

# Copyright (c) 2011 Tim Fuller

# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import os
import textwrap
import linecache
import numpy as np
import matplotlib.pyplot as plt

# x/y aspect ratio
aspect_ratio = 4.0/3.0

def info(s,n=0):
    if not verbose: return
    prefix = "  ====== INFO: " + "  "*n
    para = textwrap.wrap(s,width=72-len(prefix))
    sys.stdout.write(prefix+para[0]+"\n")
    if len(para) > 1:
        for i in range(1,len(para)):
            sys.stdout.write(len(prefix)*" "+para[i]+"\n")

def error(s):
    prefix = "  ===== ERROR: "
    para = textwrap.wrap(s,width=72-len(prefix))
    sys.stdout.write(prefix+para[0]+"\n")
    if len(para) > 1:
        for i in range(1,len(para)):
            sys.stderr.write(len(prefix)*" "+para[i]+"\n")
    sys.exit()

def postprocess(file_list,verbosity=1):
    global verbose
    verbose = verbosity > 0
    info("Checking input...")
    if type(file_list) == str:
        file_list = [file_list]

    for myfile in file_list:
        if not os.path.isfile(myfile):
            error("File not found: {0}".format(myfile))

    info("Starting Postprocessing")
    for myfile in file_list:

        info("Processing file {0}".format(myfile))
        headers = linecache.getline(myfile,1).split()
        f = open(myfile,"r")
        data = (np.loadtxt(f,skiprows=1)).transpose()
        cols, rows = np.shape(data)
        if len(headers) != cols:
            error("Number of headers does not match number of columns")
        info("{0} columns found".format(cols),n=1)

        html_f = myfile+".html"
        html_d = myfile+"_html"
        # Look for the directory and file, make it if it doesn't exist
        if not os.path.isdir(html_d):
            try:
                os.mkdir(html_d)
                info("Made directory {0}".format(html_d),n=1)
            except:
                info("Could not make ddirectory {0}".format(html_d),n=1)
        info("Outputing html to {0}".format(html_f),n=1)
        # Find the TIME header, if possible. Otherwise just use the first
        # column for the x-axis.
        idx_time = 0
        for i in range(0,cols):
            if headers[i].upper() == "TIME":
                idx_time = i
                break
        info("Using '{0}' for the abscissa".format(headers[idx_time]),n=1)
        plots = []
        for i in range(0,cols):
            if i == idx_time: continue
            plots.append(os.path.join(html_d,headers[i]+".png"))
            info("Plotting {0}".format(headers[i]),n=2)
            plt.clf() # Clear the current figure
            plt.cla() # Clear the current axes
            plt.plot(data[idx_time],data[i])
            plt.xlabel(headers[idx_time])
            plt.ylabel(headers[i])
            plt.gcf().set_size_inches(aspect_ratio*5,5.0)
            plt.savefig(plots[-1],dpi=100)

        OUTPUT = open(html_f,"w")
        OUTPUT.write("<html>\n"                      +\
                     "  <head>\n"                    +\
                     "    <title>"+myfile+"</title>\n" +\
                     "  </head>\n"                   +\
                     "  <body>\n"                    +\
                     "  <table>\n"                   +\
                     "    <tr>\n")
        for i in range(0,len(plots)):
            if i%3 == 0 and i != 0 and i != len(plots)-1:
                OUTPUT.write("    </tr>\n" +\
                             "    <tr>\n")
            width = str(int( aspect_ratio*300   ))
            height = str(int( 300   ))
            OUTPUT.write("      <td>\n"                               +\
                         "        "+os.path.basename(plots[i])+"\n"   +\
                         "        <a href=\""+plots[i]+"\">\n"        +\
                         "        <img src=\""+plots[i]+"\" "         +\
                                        "width=\""+width+"\" "        +\
                                        "height=\""+height+"\">\n"    +\
                         "        </a>\n"                             +\
                         "      </td>\n")


        OUTPUT.write("      </tr>\n    </table>\n  </body>\n</html>\n")
        OUTPUT.close()

if __name__ == '__main__':
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        sys.exit(
            "Usage:\n"                                              +\
            "    {0} file1.txt file2.out ...\n".format(sys.argv[0]) +\
            "Generates an HTML file and folder of variable plots\n" +\
            "contained in the whitespace-delimited text file\n")
    postprocess(sys.argv[1:])
