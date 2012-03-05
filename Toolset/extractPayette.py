#!/usr/bin/env python

import sys,os
import numpy
import optparse

"""
NAME
    extractPayette.py

PURPOSE
    extract selected columns of data from a Payette simulation output file

INPUT
    file name

OUTPUT
    basename.xout
"""

exe = os.path.basename(__file__)
manpage=\
"""
NAME
      {0} - extract data from a Payette simulation output file

SYNOPSIS
      {0} [options] <file name> @keywords %columns

DESCRIPTION
      The {0} script is designed to parse, extract, and operate on data from
      a Payette simulation output file, though it can be used on any similarly
      formatted text file.

      By default, output is printed to the console.  Using the --xout option,
      output is printed to the console and written to a file.

EXAMPLES
      Extract and print columns 1 and 5 from output.out
        % {0} output.out %1 %5

      Extract and print columns 1,2 and add columns 3 and 4 from output.out
        % {0} output.out %1 %2 %3+%4

      Extract and print columns time, eps11, sig11, and sig11+sig22+sig33
        % {0} output.out @time @eps11 @sig11 @sig11+@sig22+@sig33
""".format(exe)

kwtoken, coltoken = "@", "%"

optokens = [ "+", "-", "*", "/", "**", "^" ]
tokens = optokens + [kwtoken] + [coltoken]
opchoices = ["","print"]

def main(argv):

    def col2int(col):
        if col[0] == coltoken: return int(col[1:])
        else: error("bad column specifier {0} sent to col2int".format(col))
        return

    usage =\
"""
{0}: extract columns of data from a Payette simulation output file.
  % {0} [options] file [@key1 [@key2 [...]]] [%col1 [%col2 [ %coln ]]]""".format(exe)

    parser = optparse.OptionParser(usage = usage, version = "%prog 1.0")
    parser.add_option("-s","--sep",dest="SEP",action="store",type="choice",
                      choices=("space","tab","comma"),default="space",
                      help=("output file column seperation format. "
                            "[choices: (space, tab, comma) ] [default: space]"))
    parser.add_option("--cols",dest="COLS",action="store_true", default=False,
                      help="print data column numbers [default: %default]")
    parser.add_option("--man",dest="MAN",action="store_true",default=False,
                      help="print man page [default: %default]")
    parser.add_option("--xout",dest="XOUT",action="store_true",default=False,
                      help=("write results to file 'input_file.xout'"
                            " [default: %default]"))
    parser.add_option("--operation",dest="OPP",action="store",default=None,
                      type="choice",choices=(None,"print"),
                      help=("Operation to perform, choose from ({0})"
                            .format(", ".join(opchoices))))
    (opts, passed_args) = parser.parse_args(argv)
    if opts.MAN:
        print manpage
        parser.print_help()
        return 0

    if not len(passed_args):
        parser.print_help()
        parser.error("Must specify file name")
        pass

    sep = {"space":" ", "tab":"\t", "comma":","}[opts.SEP]

    args = args2dict(passed_args,sep)

    # args is now a list of the form
    # [{"file":file_1,"extract":[things to extract],"extract header":"header str"},
    #  {"file":file_2,"extract":[things to extract],"extract header":"header str"},
    #                                 .
    #                                 .
    #                                 .
    # ]

    # parse the lists and perform operations
    for arg in args:
        outf = arg["file"]
        (fnam, fext) = os.path.splitext(outf)
        header = arg["extract header"]
        to_extract = arg["extract"]

        # only print out column names
        if opts.COLS:
            print_cols(fnam,arg["file header"])
            continue

        # open the file and extract only the data that the user asked for
        data = []
        for line in open(outf,"r").readlines()[1:]:

            if not line.split(): continue

            linedat = [ float(x) for x in line.split() ]

            extracted_data = []
            for item in to_extract:
                oper = isinstance(item,(list,tuple))

                if not oper:
                    # extract requested column
                    extracted_data.append(linedat[col2int(item)])
                    continue

                # user specified an operation to perform
                operation = []
                for x in item:
                    if x[0] == coltoken:
                        operation.append(str(linedat[col2int(x)]))
                    else:
                        operation.append(x)
                        pass
                    continue
                extracted_data.append(eval("".join(operation)))

                continue

            data.append(extracted_data)

            continue

        xout = fnam + ".xout" if opts.XOUT else None
        logger = Logger(xout,"w")
        logger.write(header)
        for datline in data:
            # write out data to file
            logger.write(sep.join([ffrmt(x) for x in datline]))
            continue
        del logger

        # for any further processing, convert to numpy array
        # data = numpy.array(data)

        continue

    return

def ffrmt(x):
    return "{0:12.5E}".format(x)

def message(msg):
    sys.stdout.write("{0}: INFO: {1}\n".format(exe, msg))

def error(msg):
    sys.exit("{0}: ERROR: {1}".format(exe, msg))

def args2dict(args,sep):

    """
    PURPOSE
        parse args for output files and options.  Gather operations into list and
        replace column keywords with the column numbers.

    INPUT
        args: the list of user specified command line arguments

    OUTPUT
        parsed_args: [ [file1,%i,%j,...,[op1],[op2],...]
                       [file2,%i,%j,...,[op1],[op2],...]
                                         .
                                         .
                                         .
                       [filen,%i,%j,...,[op1],[op2],...] ]
    """

    def check_file():
        length = 0
        for iline,line in enumerate(open(argf,"r").readlines()[1:]):
            if not line.split(): continue
            linedat = [ float(x) for x in line.split() ]
            if iline == 0: length = len(linedat)
            else:
                if len(linedat) != length:
                    error("Number of columns in line {0} of {1} not consistent"
                          .format(iline+1,argf))
                    pass
                pass
            continue
        return

    def kw2col(kw):
        nam = kw
        try: col = coltoken + str(head_dict[kw.lower()])
        except: error("keyword {0} not in {1}, choose from: {2}"
                      .format(x,argf,header))
        return col, kw

    def col2col(col):
        """ return true column number (requested - 1) and column name """
        try:
            col = int(col) - 1
            kw = header.split()[col]
        except ValueError:
            error("non integer column number {0}".format(col))
        except IndexError:
            error("{0} has only {1} columns, requested column {2}"
                  .format(argf,len(header.split()),col+1))
        except:
            error("error processing {0} in {1}" .format(arg,args))
            pass
        col = coltoken + str(col)
        return col, kw

    iarg = 0
    parsed_args = []
    while iarg < len(args):

        argf = args[iarg]

        if not os.path.isfile(argf):
            error("Expected valid file, got {0}".format(argf))
            pass

        # check if same file is repeated
        if [True for x in parsed_args if args[iarg] in x]:
            error("file {0} sent multiple times".format(argf))
            pass

        # add file to tmparg and move on to next item in args
        check_file()
        arg_dict = {}
        arg_dict["file"] = argf
        arg_dict["extract"] = []
        arg_dict["extract header"] = []
        fnam, fext =  os.path.splitext(argf)
        header = open(argf).readline()
        head_dict = header2dict(header)
        arg_dict["file header"] = header.strip()

        iarg += 1

        # interegate args from the last valid file to the next valid file
        while iarg < len(args):

            arg = args[iarg]

            if os.path.isfile(arg):
                # ran in to another file, continue on to next
                break

            if [x for x in arg if x in optokens]:
                # column operation has been requested, checkit

                if len(arg) == 1:
                    # optoken on its own not allowed
                    bad_op(arg,args)

                elif arg[0] in optokens:
                    # beginning with optoken is ambiguous
                    ambiguous_op(arg,args)

                elif arg[-1] in optokens:
                    # ending with optoken is ambiguous
                    ambiguous_op(arg,args)

                else:
                    pass

                # split arg at all optokens and replace keywords with columns,
                # get column names
                tmp = ""
                for ii in arg:
                    if ii in optokens: ii = " {0:s} ".format(ii)
                    tmp += ii
                    continue
                tmpcol,tmpnam = [],[]
                tmparg = [x for x in tmp.split(" ") if x]
                for ix,x in enumerate(tmparg):
                    if x[0] == kwtoken:
                        if len(x) == 1:
                            error("empty keyword identifier".format(args))
                        tmparg[ix],nam = kw2col(x[1:])
                    elif x[0] == coltoken:
                        if len(x) == 1:
                            error("empty column identifier".format(args))
                        tmparg[ix],nam = col2col(x[1:])
                    else:
                        nam = x
                        pass
                    tmpnam.append(nam)

                    continue
                arg_dict["extract header"].append("".join([x for x in tmpnam if x]))
                arg_dict["extract"].append(tmparg)

            else:

                if arg[0] == kwtoken:
                    if len(arg) == 1:
                        error("empty keyword identifier".format(args))
                        pass
                    arg,nam = kw2col(arg[1:])
                elif arg[0] == coltoken:
                    if len(arg) == 1:
                        error("empty column identifier".format(args))
                        pass
                    arg,nam = col2col(arg[1:])
                else:
                    nam = arg
                    pass
                arg_dict["extract header"].append(nam)
                arg_dict["extract"].append(arg)
                pass

            iarg += 1
            continue

        # check that all extraction requests are valid
        for item in arg_dict["extract"]:
            if isinstance(item,(list,tuple)):
                xval = "".join([x.replace(coltoken,"") for x in item])
            else:
                xval = item.replace(coltoken,"")
            try: eval("".join([x.replace(coltoken,"") for x in item]))
            except: error("bad extraction request: {0}".format(item))
            continue

        # header is now a list, join it with the user requested separation
        arg_dict["extract header"] = sep.join(arg_dict["extract header"])
        parsed_args.append(arg_dict)

        continue

    return parsed_args

def header2dict(header):
    dicthead = {}
    for i,item in enumerate(header.split()):
        dicthead[item.lower()] = i
        continue
    return dicthead

def bad_op(op,args):
    sys.exit("""bad operation specification "{0}" in "{1}".
Operations between entries must not be padded with any whitespace in the
argument list.  i.e., specify @kw1+@kw2 and not @kw1 + @kw2"""
             .format(op," ".join(args)))
    return

def ambiguous_op(op,args):
    sys.exit("ambiguous operation specification '{0}' in '{1}'"
             .format(op," ".join(args)))
    return

def arg2list(arg):
    if not isinstance(arg, (tuple,list)):
        return [arg]

    flattened = []
    for item in arg:
        if isinstance(item, (tuple,list)):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
            pass
        continue

    return flattened


def print_cols(fnam,header):
    if not isinstance(header,list): header = header.split()
    with open(fnam + ".cols", "w") as colf:
        for i, col in enumerate(header):
            item = "{0:d}\t{1:s}\n".format(i+1, col)
            sys.stdout.write(item)
            colf.write(item)
            continue
        pass
    return

class Logger(object):
    def __init__(self,name,mode):
        self.stdout = sys.stdout
        if name: self.file = open(name,mode)
        else: self.file = None
        sys.stdout = self
        pass
    def __del__(self):
        sys.stdout = self.stdout
        if self.file: self.file.close()
        pass
    def write(self,data):
        if self.file: self.file.write(data + "\n")
        self.stdout.write(data + "\n")

if __name__ == "__main__":
    main(sys.argv[1:])
