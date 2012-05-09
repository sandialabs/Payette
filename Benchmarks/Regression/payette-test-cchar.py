#!/usr/bin/env python
from Payette_config import *
from Source.Payette_test import PayetteTest

class Test(PayetteTest):

    def __init__(self, check=True):


        self.enabled = True

        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.tdir = os.path.dirname(os.path.realpath(__file__))

        self.infile = "{0}.inp".format(os.path.join(self.tdir,self.name))
        self.outfile = "{0}.out".format(self.name)
        self.runcommand = ["runPayette","--no-writeprops","--no-restart",
                           "--cchar=!",self.infile]
        self.material = "elastic"
        self.keywords = ["payette","cchar","regression","fast"] # add keywords

        self.owner = "Tim Fuller"
        self.date = "February 25, 2012"
        self.description = """ Test of user comment character capabilities """

        if check:
            self.check_setup()

        pass

    def runTest(self):
        """ run the test """

        perform_calcs = self.run_command(self.runcommand)

        if perform_calcs != 0:
            return self.failcode

        return self.passcode

if __name__ == "__main__":
    import time

    test = Test()
    if "--cleanup" in sys.argv:
        for ext in ["out","res","log","prf","pyc","echo"]:
            try: os.remove("%s.%s"%(test.name,ext))
            except: pass
            continue
        pass
    else:
        t0 = time.time()
        print("%s RUNNING"%test.name)
        run_test = test.runTest()
        dtp = time.time()-t0
        if run_test == test.passcode:
            print("%s PASSED(%fs)"%(test.name,dtp))
        elif run_test == test.diffcode:
            print("%s DIFFED(%fs)"%(test.name,dtp))
        else:
            print("%s FAILED(%fs)"%(test.name,dtp))
