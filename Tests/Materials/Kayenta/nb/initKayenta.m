(* ::Package:: *)

(************************************************************************)
(* This file was generated automatically by the Mathematica front end.  *)
(* It contains Initialization cells from a Notebook file, which         *)
(* typically will have the same name as this file except ending in      *)
(* ".nb" instead of ".m".                                               *)
(*                                                                      *)
(* This file is intended to be loaded into the Mathematica kernel using *)
(* the package loading commands Get or Needs.  Doing so is equivalent   *)
(* to using the Evaluate Initialization Cells menu command in the front *)
(* end.                                                                 *)
(*                                                                      *)
(* DO NOT EDIT THIS FILE.  This entire file is regenerated              *)
(* automatically each time the parent Notebook file is saved in the     *)
(* Mathematica front end.  Any changes you make to this file will be    *)
(* overwritten.                                                         *)
(************************************************************************)



ClearAll["Global`*"]


demodir=NotebookDirectory[]
(*demodir="$DEMODIR"*)


initialized={};
init[varNameString_,varDescriptionString_,varValue_]:=Module[{},initialized=Join[initialized,{{varNameString<>"=",varDescriptionString<>"=",varValue}}];MatrixForm[{initialized[[-1]]}]]

initlist:=Sort[initialized]//MatrixForm


PHONY=-0.123456789*^9
init["PHONY","a fake initialization parameter",PHONY];


SetDirectory[NotebookDirectory[]]


rundir=NotebookDirectory[]
(*rundir="$RUNDIR"*)
init["rundir","directory in which demo program was run",rundir];


SetDirectory[rundir]


MakeGraphs=False;
init["MakeGraphs","logical indicating if sim graphics should be exported",MakeGraphs];


grafhis[x_,string_]:=If[MakeGraphs,Export[demodir<>string<>".gif",showhis[x,string]],showhis[x,string]]
init["grafhis","function for exporting history plots","grafhis[...]"];


xport[name_,grafik_]:=If[MakeGraphs,Export[name,grafik],"MakeGraphs is False"]


MPa=10^6;
init["MPa","megapascals per SI unit of pressure (pascal)",MPa]


red=RGBColor[1,0,0];green=RGBColor[0,1,0];blue=RGBColor[0,0,1];RGBplotstyle:={red,green,blue};fatLine=Thickness[0.02];fatterLine=Thickness[0.03];medLine=Thickness[0.01];thinLine=Thickness[0.005];
init["red","personally defined color",red];
init["green","personally defined color",green];
init["blue","personally defined color",blue];
init["RGBplotstyle","style useful for overlaid plots","{red,green,blue}"];
init["fatLine","a bold line thickness",fatLine];
init["fatterLine","a REALLY bold line thickness",fatterLine];
init["medLine","a moderate line thickness",medLine];
init["thinLine","a thin line thickness",thinLine];

chainDash=Dashing[{0.005,0.02,0.02,0.02}];
myColor=RGBColor[.2,0,.6];
myThickness=Thickness[0.01];
myStyle:={myColor,myThickness};
init["chainDash","a format style for chaindashed lines",chainDash];
init["myColor","my favorite color for some plots",myColor];
init["myThickness","my favorite thickness for some plots",myThickness];
init["myStyle","combined favorite style for plots","{myColor,myThickness}"];



if[logical_]:=If[logical,1,0]
init["if[logical]","function that returns 1 if 'logical' is True, or 0 otherwise","if[...]"]


numCol=3;
init["numCol","number of columns in quick plot graphics arrays",numCol];


setRun[runID_]:=(
runid=runID;
demodir=ToFileName[{rundir,runid}];

math1=ToFileName[demodir,runid<>".math1"];
math2=ToFileName[demodir,runid<>".math2"];
runlog=ToFileName[demodir,runid<>".log"];
runprops=ToFileName[demodir,runid<>".props"];
ReadList[math1];

If[NEWXSV,solveKapFromX[X]];
If[NEWXSV,solveQSELFromX[X]];
plotz=ReadList[math2];
plotvarz=ReadList["!grep -h plotable "<>runlog,String];

propz=ReadList["!cat "<>runprops,String];
computedVARS[];

(* since payette doesn't track USM like the demo driver, set it to the constant initial value *)
USM=ConstantArray[B0+4/3*G0,lastep];

Print[TableForm[{Definition[demodir],Definition[math1],Definition[math2]}]];
Grid[Partition[plotz,numCol,numCol,1,{}]]
)
init["setRun","command that reads in results from a particular MED problem","setRun[runID]"];


plotable:=(plotvarz//TableForm);
plottable:=plotable;
init["plotable","type this to see keywords of plottable variables",numCol];

props:=Grid[Partition[Sort[propz],4,4,1," "],Frame->All,Alignment->Left];
init["props","type props to see explicitly specified material properties",numCol];



maketab[x_,y_]:=Table[{x[[i]],y[[i]]},{i,1,Length[x]}];
init["maketab","creates a table of {x,y} pairs","maketab[x,y]"];


interp[tab_]:=Interpolation[tab,InterpolationOrder->1]
interp[x_,y_]:=Interpolation[maketab[x,y],InterpolationOrder->1]
init["interp","creates an interpolation function from discrete data","interp[xydataTable] or interp[xlist,ylist]"];


showcy[x_]:=ListPlot[Table[{i,x[[i]]},{i,Length[x]}],Joined->True,PlotStyle->myStyle]
init["showcy","command to plot a variable vs. dump cycle number","showcy[...]"];


showcy[x_,labl_]:=ListPlot[Table[{i,x[[i]]},{i,Length[x]}],Joined->True,AxesLabel->labl,GridLines->Automatic,AxesOrigin->{1,0},PlotStyle->myStyle]


showcy[x_,imin_,imax_]:=ListPlot[Table[{i,x[[i]]},{i,imin,imax}],Joined->True,PlotStyle->myStyle]


showhis[y_]:=ListPlot[Transpose[{time,y}],Joined->True,PlotStyle->myStyle]
init["showhis","command to plot a variable vs. time","showhis[...]"];


showhis[y_,labl_]:=ListPlot[Transpose[{time,y}],Joined->True,AxesLabel->{"time",labl},PlotStyle->myStyle]


showme[x_,y_]:=ListPlot[Transpose[{x,y}],Joined->True,PlotStyle->myStyle]
init["showme","plot one variable vs. another variable","showme[x,y]"];


makeme[x_,y_]:=ListPlot[maketab[x,y],Joined->True]
init["makeme","same as showme except plot output is suppressed","makeme[...]"];


makeme[x_,y_,style_]:=ListPlot[maketab[x,y],Joined->True,PlotStyle->style]


showmeAll[x_,y_]:=ListPlot[maketab[x,y],Joined->True,PlotRange->All,PlotStyle->myStyle]
showmeAll[x_]:=ListPlot[x,Joined->True,PlotRange->All,PlotStyle->myStyle]
init["showmeAll","forces all data to be plotted","showmeAll[x,y]"];


showme[x_,y_,labl_]:=ListPlot[Transpose[{x,y}],Joined->True,AxesLabel->{labl[[1]],labl[[2]]},PlotStyle->myStyle]


showme[x_,y_,imin_,imax_]:=ListPlot[Table[{x[[i]],y[[i]]},{i,imin,imax}],Joined->True]


showme[x_,y_,imin_,imax_,labl_]:=ListPlot[Table[{x[[i]],y[[i]]},{i,imin,imax}],Joined->True,AxesLabel->{labl[[1]],labl[[2]]},PlotStyle->myStyle]


showmany[x_,yyy_]:=Module[{n},
n=Length[yyy];
Show[Table[
ListPlot[
Transpose[{x,yyy[[i]]}],
Joined->True,
PlotStyle->{ Thick,ColorData["Rainbow"][(i-1)/(n-1)] }
]
,{i,Length[yyy]}]]
]
init["showmany","plot many variables together","showmany[x,{y1,y2}]"]


showme[x_]:=showcy[x];


showdat[x_,y_]:=ListPlot[Table[{x[[i]],y[[i]]},{i,1,Length[x]}],Joined->True]
showdat[x_,y_,imin_,imax_]:=ListPlot[Table[{x[[i]],y[[i]]},{i,imin,imax}],Joined->True]
showdat[x_]:=ListPlot[x,Joined->True,AxesLabel->labl,GridLines->Automatic,AxesOrigin->{1,0}]
init["showdat","graph data tables of length differing from code-generated data","showdat[...]"];


resize[a_,n_]:=Module[{na,fac,id,idl},
na=Length[a];
fac=(na-1)/(n-1);
N[Flatten[{Table[(id=(k-1)fac+1;idl=IntegerPart[id];a[[idl]]+(a[[idl+1]]-a[[idl]])FractionalPart[id]),{k,1,n-1}],{a[[na]]}}]]
];
init["resize","interpolate an array to change its number of data points","resize[array,npts]"];


stressSpace[]:=Module[{data,mn,mx,axis,adat,xdat,ydat,zdat,val,vuctr},data=Transpose[{sig11,sig22,sig33}/(scaler)];
mn=Min[data];
mx=Max[data];
axis=Table[val,{val,mn,mx,(mx-mn)/100}];
adat=Transpose[{axis,axis,axis}];
xdat=Table[{val,0,0},{val,mn,mx,(mx-mn)/100}];
ydat=Table[{0,val,0},{val,mn,mx,(mx-mn)/100}];
zdat=Table[{0,0,val},{val,mn,mx,(mx-mn)/100}];
vuctr=({mn,mn,mn}+{mx,mx,mx})/2;
ListPointPlot3D[{adat,xdat,ydat,zdat,data},BoxRatios->{1,1,1},PlotStyle->{Black,Red,Green,Blue,Orange}]
]
init["stressSpace","show path through 3D stress space","stressSpace[]"];


RKfnt[RKARG_,I1_,a2_,a3_,a4_]:=Module[{rk,rkmin},rkmin=If[J3TYPEM==1,0.778,0.49999];rk=Max[rkmin, If[RKARG>0,RKARG,1/(1+Sqrt[3](a4+a2 a3 E^(a2 I1)))]   ]
]
init["RKfnt","function for pressure varying strength ratio","RKfnt[...]"];


RK0:=RKfnt[RKM,0,A2M,A3M,A4M]
RKPF0:=RKfnt[RKPFM,0,A2PFM,A3M,A4PFM]
init["RK0","value of pressure varying strength ratio at zero pressure",RK0];


gudehus[b_,RK_]:=(1/2) (1+Sin[3b]+(1/RK)(1-Sin[3b]));
init["gudehus","The Gudehus third invariant function","gudehus[...]"];

willamWarnke[b_,RK_]:=(4(1-RK^2)Cos[Pi/6+b]^2+(2 RK-1)^2)/(2(1-RK^2)Cos[Pi/6+b]+(2 RK-1)Sqrt[4(1-RK^2)Cos[Pi/6+b]^2+5RK^2-4RK]);
init["willamWarnke","The Willam-Warnke third invariant function","willamWarnke[...]"];

mohrCoulomb[b_,RK_]:=Module[{SI,scale},SI=3(1-RK)/(1+RK);
scale=2 Sqrt[3]/(3-SI);
(Cos[b]-Sin[b]*SI/Sqrt[3])scale];
init["mohrCoulomb","The Mohr-Coulomb third invariant function","mohrCoulomb[...]"];

GAM[J3TYPE_,RK_,b_]:=Switch[J3TYPE,1,gudehus[b,RK],2,willamWarnke[b,RK],3,mohrCoulomb[b,RK]]
GAM[J3TYPE_,b_]:=Switch[J3TYPE,1,gudehus[b,RK0],2,willamWarnke[b,RK0],3,mohrCoulomb[b,RK0]]
init["GAM","Fossum's GAMMA function for third-invariant dependence","GAM[...]"];


octo[scale_]:=Module[{bfnt,FGAMM,Frfnt,Fradius,Fstyle,GGAMM,Grfnt,Gradius,Gstyle,octoPlot},bfnt[t_]:=(1/3)ArcSin[Sin[3t]];
FGAMM[b_]:=Evaluate[GAM[IntegerPart[J3TYPEM],RK0,b]];
GGAMM[b_]:=Evaluate[GAM[IntegerPart[J3TYPEM],RKPF0,b]];
Frfnt[b_]:=Evaluate[1/FGAMM[b]];
Fradius[t_]:=Evaluate[Frfnt[bfnt[t]]];
Grfnt[b_]:=Evaluate[1/GGAMM[b]];
Gradius[t_]:=Evaluate[Grfnt[bfnt[t]]];
Fstyle={medLine,RGBColor[0,0,.9]};
Gstyle={thinLine,RGBColor[.8,.8,.9]};
octoPlot[size_,radius_,style_]:=ParametricPlot[{size radius[t]Cos[t],size radius[t] Sin[t]},{t,0,2Pi},AspectRatio->Automatic,Axes->False,PlotStyle->style];
Show[ octoPlot[rmax/scale,Fradius,Fstyle],octoPlot[rmax/scale,Gradius,Gstyle] ]
]
init["octo","function to create an image of the octahedral profile","octo[scale,I1]"];


octo[scale_,I1_]:=Module[{RKatI1,RKPFatI1,bfnt,FGAMM,Frfnt,Fradius,Fstyle,GGAMM,Grfnt,Gradius,Gstyle,octoPlot},
RKatI1=RKfnt[RKM,I1,A2M,A3M,A4M];
RKPFatI1=RKfnt[RKPFM,I1,A2PFM,A3M,A4PFM];
bfnt[t_]:=(1/3)ArcSin[Sin[3t]];
FGAMM[b_]:=Evaluate[GAM[IntegerPart[J3TYPEM],RKatI1,b]];
GGAMM[b_]:=Evaluate[GAM[IntegerPart[J3TYPEM],RKPFatI1,b]];
Frfnt[b_]:=Evaluate[1/FGAMM[b]];
Fradius[t_]:=Evaluate[Frfnt[bfnt[t]]];
Grfnt[b_]:=Evaluate[1/GGAMM[b]];
Gradius[t_]:=Evaluate[Grfnt[bfnt[t]]];
Fstyle={medLine,RGBColor[0,0,.9]};
Gstyle={thinLine,RGBColor[.8,.8,.9]};
octoPlot[size_,radius_,style_]:=ParametricPlot[{size radius[t]Cos[t],size radius[t] Sin[t]},{t,0,2Pi},AspectRatio->Automatic,Axes->False,PlotStyle->style];
Show[octoPlot[rmax/scale,Fradius,Fstyle],octoPlot[rmax/scale,Gradius,Gstyle]]]


atan[x_,y_]:=ArcTan[x,y]
atan[0,0]:=0
atan[0.,0.]:=0.;
LodeX[Txx_,Tyy_,Tzz_]:=(Tyy-Txx)/Sqrt[2]
init["LodeX","returns x-Lode coordinate for true triaxial loading","LodeX[xx,yy,zz]"];
LodeY[Txx_,Tyy_,Tzz_]:=(2Tzz-Txx-Tyy)/Sqrt[6]
init["LodeY","returns y-Lode coordinate for true triaxial loading","LodeY[xx,yy,zz]"];
TrueLode[TlodeX_,TlodeY_]:=Table[atan[TlodeX[[i]],TlodeY[[i]]](180/Pi),{i,1,Length[TlodeX]}]
init["TrueLode","returns the full-range Lode angle for triaxial loading","TrueLode[TlodeX,TlodeY]"];


OCTO1[istep_]:=Module[{RKatI1,RKPFatI1,bfnt,FGAMM,Frfnt,Fradius,Fstyle,range,octoPlot},range=1.1` rmax;i1=I1[[istep]];RKatI1=RKfnt[RKM,i1,A2M,A3M,A4M];bfnt[t_]:=1/3 ArcSin[Sin[3 t]];FGAMM[b_]:=Evaluate[GAM[IntegerPart[J3TYPEM],RKatI1,b]];Frfnt[b_]:=Evaluate[1/FGAMM[b]];Fradius[t_]:=Evaluate[Frfnt[bfnt[t]]];Fstyle={thinLine,RGBColor[0,1,0]};octoPlot[size_,radius_,ctr_,style_]:=ParametricPlot[{size radius[t] Cos[t]+ctr[[1]],size radius[t] Sin[t]+ctr[[2]]},{t,0,2 \[Pi]},PlotRange->{{-range,range},{-range,range}},AspectRatio->Automatic,Axes->False,PlotStyle->style,ImageSize->2 72];center={LodeXbs[[istep]],LodeYbs[[istep]]};jnk0=octoPlot[Sqrt[2] A1M,Fradius,{0,0},{medLine,RGBColor[0,0,1]}];jnk1=octoPlot[Sqrt[2] (A1M-RNM),Fradius,center,Fstyle];jnk2=ListPlot[Table[{LodeXsig[[i]],LodeYsig[[i]]},{i,1,istep}],Joined->True,PlotRange->{{-range,range},{-range,range}},AspectRatio->Automatic,ImageSize->2 72,Axes->False,PlotStyle->{fatterLine}];Show[jnk2,jnk1,jnk0,AspectRatio->Automatic]]


OCTO1[istep_]:=Module[{i1,RKatI1,RKPFatI1,bfnt,FGAMM,Frfnt,Fradius,Fstyle,octoPlot,jnk0,jnk1,jnk2},range=1.1` rmax;i1=I1[[istep]];RKatI1=RKfnt[RKM,I1,A2M,A3M,A4M];bfnt[t_]:=1/3 ArcSin[Sin[3 t]];FGAMM[b_]:=Evaluate[GAM[IntegerPart[J3TYPEM],RKatI1,b]];Frfnt[b_]:=Evaluate[1/FGAMM[b]];Fradius[t_]:=Evaluate[Frfnt[bfnt[t]]];Fstyle={medLine,RGBColor[0,1,0]};Gstyle={thinLine,RGBColor[0,1,0]};octoPlot[size_,radius_,style_]:=ParametricPlot[{size radius[t] Cos[t],size radius[t] Sin[t]},{t,0,2 \[Pi]},PlotRange->{{-range,range},{-range,range}},AspectRatio->Automatic,Axes->False,PlotStyle->style];octoPlot[size_,radius_,ctr_,style_]:=ParametricPlot[{size radius[t] Cos[t]+ctr[[1]],size radius[t] Sin[t]+ctr[[2]]},{t,0,2 \[Pi]},PlotRange->{{-range,range},{-range,range}},AspectRatio->Automatic,Axes->False,PlotStyle->style,ImageSize->2 72];center={LodeXbs[[istep]],LodeYbs[[istep]]};jnk0=octoPlot[Sqrt[2] Ff[-i1,A1M,A2M,A3M,A4M],Fradius,{0,0},{medLine,RGBColor[0,0,1]}];jnk1=octoPlot[Sqrt[2] (Ff[-i1,A1M,A2M,A3M,A4M]-RNM) Sqrt[Fc[-i1,-KAPPA[[istep]],CRM,A1M,A2M,A3M,A4M]],Fradius,center,Fstyle];jnk2=ListPlot[Table[{LodeXsig[[i]],LodeYsig[[i]]},{i,1,istep}],Joined->True,PlotRange->{{-range,range},{-range,range}},AspectRatio->Automatic,ImageSize->2 72,Axes->False,PlotStyle->{fatterLine}];Show[jnk0,jnk1,jnk2]];
init["OCTO1","draws yield and limit octahedral profiles at a specified timestep","OCTO1[istep]"];


octoLOHI[scale_]:=If[MakeGraphs,(Export[demodir<>"octoLO.gif",octo[scale,Max[I1]]];Export[demodir<>"octoHI.gif",octo[scale,Min[I1]]]),(octo[scale,Max[I1]];octo[scale,Min[I1]])]
init["octoLOHI","plots the octahedral profiles at the max and min pressures","octoLOHI[scale]"];


sign[x_,y_]:=Abs[x] (2 UnitStep[y]-1)
init["sign","transfers the numerical sign (pos or neg) of y to x","sign[x,y]"];


eunderflow=-34.53877639491;eoverflow=92.1034037;exps[x_]:=E^(Min[Max[x,eunderflow],eoverflow])
Exps[a_]:=Table[exps[a[[i]]],{i,1,Length[a]}]
init["exps","identical to the exponential function with over/underflow cut-offs","exps[x]"];


Ff[I1bar_,a1_,a2_,a3_,a4_]:=a1-a3 E^(-a2 I1bar)+a4 I1bar;
init["Ff","function for the limit surface","Ff[...]"];
ff[I1bar_,a1_,a2_,a3_,a4_,N_]:=Ff[I1bar,a1,a2,a3,a4]-N;
init["ff","same as Ff, but without the kinematic hardening shift","ff"];
Xbar[kappaBar_,CR_,a1_,a2_,a3_,a4_]:=kappaBar+CR Ff[kappaBar,a1,a2,a3,a4]
init["Xbar","Fossum's X function for I1 vs. volumetric plastic strain","Xbar"];
Fc[I1bar_,kappaBar_,CR_,a1_,a2_,a3_,a4_]:=Max[0,1-(I1bar-kappaBar)(Abs[I1bar-kappaBar]+(I1bar-kappaBar))/(2(Xbar[kappaBar,CR,a1,a2,a3,a4]-kappaBar)^2)];
init["Fc","Fossum's Fc function (for the elliptical cap)","Fc"];
fc[I1bar_,kappaBar_,CR_,a1_,a2_,a3_,a4_]:=Sqrt[Fc[I1bar,kappaBar,CR,a1,a2,a3,a4]];
init["fc","Square root of the Fc function","fc"];
rootJ2merid[I1bar_,a1_,a2_,a3_,a4_,N_,kappaBar_,CR_]:=(Ff[I1bar,a1,a2,a3,a4]-N)fc[I1bar,kappaBar,CR,a1,a2,a3,a4];
init["rootJ2merid","function for Sqrt[J2] vs. I1 at yield","rootJ2merid[...]"];
merid[zbar_,a1_,a2_,a3_,a4_,N_,kappaBar_,CR_]:=(Sqrt[2]) rootJ2merid[(Sqrt[3]) zbar,a1,a2,a3,a4,N,kappaBar,CR]
init["merid","function for Lode radius vs. Lode z at yield","merid[...]"];
limitCurve[zbar_,a1_,a2_,a3_,a4_]:=Sqrt[2]Ff[Sqrt[3]zbar,a1,a2,a3,a4]
init["limitCurve","function for Lode radius vs. Lode z at limit","limitCurve[...]"];
limitCurveTXE[RK0_,zbar_,a1_,a2_,a3_,a4_]:=Sqrt[2]Ff[Sqrt[3]zbar,a1,a2,a3,a4] RKfnt[RKM,-Sqrt[3]zbar,a2,a3,a4];
init["limitCurveTXE","same as limitCurve except TXE","limitCurveTXE[...]"];


EVP[XbarArg_]:=P3M (exps[P1M(-XbarArg-P0M)-P2M (-XbarArg-P0M)^2]-1)
init["EVP","returns plastic volumetric strain in terms of I1 for the cap","EVP[...]"];


indexAbsolute=1;indexSector=2;
indexCartesian=1;indexPolar=2;
indexR=1;indexT=2;indexZ=3;
indexX=1;indexY=2;


LodeCoords[a1_,a2_,a3_]:=Module[{x,y,z,truex,truey,radius,trueLode,Lode},
truex=(a3-a1)/Sqrt[2];
truey=-(a1+a3-2a2)/Sqrt[6];
z=-(a1+a2+a3)/Sqrt[3];
radius=Simplify[Sqrt[truex^2+truey^2]];
trueLode=If[truex==0&&truey==0,0,Simplify[ArcTan[truex,truey]]];
Lode=(1/3)ArcSin[Sin[3trueLode]];
x=radius Cos[Lode];
y=radius Sin[Lode];
{{{truex,truey,z},{radius,trueLode,z}},{{x,y,z},{radius,Lode,z}}}]
init["LodeCoords","returns lode Cartesian and cylindrical coordinates","LodeCoords[...]"];


computedVARS[]:=Module[{},

xpropz[geocrack=Round[JOBFAILU]!=0,"geocrack"];

If[geocrack,xpropz[A1U=A1M=STRENIU,"A1U"];
xpropz[A2U=A2M=FSLOPEIU/STRENIU,"A2U"];
xpropz[A3U=A3M=STRENIU E^(-((FSLOPEIU PEAKI1IU)/STRENIU)),"A3U"];];

xplotz[pres=-(1/3) (sig11+sig22+sig33),"pres","pressure = - tr(stress)/3"];xplotz[vstrain=eps11+eps22+eps33,"vstrain","volumetric strain"];
xplotz[sigm=-(I1/Sqrt[3]),"sigm","isomorphic mean stress = -I1/Sqrt[3]"];
xplotz[sigd11=sig11+PRES,"sigd11","11 component of stress deviator"];
xplotz[sigd22=sig22+PRES,"sigd22","22 component of stress deviator"];
xplotz[sigd33=sig33+PRES,"sigd33","33 component of stress deviator"];xplotz[sigs=Sqrt[sigd11 sigd11+sigd22 sigd22+sigd33 sigd33] ,"sigs","isomorphic shear stress"];

sigmrange={Min[sigm],Max[sigm]};
sigsrange={Min[sigs (2UnitStep[LODE]-1)],Max[sigs (2UnitStep[LODE]-1)]};
xpropz[punyEPS=0.0001,"punyEPS"];
punyI1=3 (B0M+B1M/2) punyEPS;
xpropz[punyI1,"punyI1"];
scaleRJ2=A1M-A3M E^(-A2M punyI1)+A4M punyI1;
xpropz[scaleI1=Abs[P0M],"scaleI1"];
xpropz[scalez=scaleI1/Sqrt[3],"scaleI1"];
scaler=scaleRJ2 Sqrt[2];If[scaler>2 G0M 1000,scaler=2 G0M;scaleRJ2=scaler/Sqrt[2]];
xpropz[scaler,"scaler"];
xpropz[scaleRJ2,"scaleRJ2"];
xpropz[j3type=IntegerPart[J3TYPEM],"j3type"];xplotz[RK=Table[RKfnt[RKM,I1[[i]],A2M,A3M,A4M],{i,Length[I1]}],"RK","value of the RK function"];
xplotz[RKPF=Table[RKfnt[RKPFM,I1[[i]],A2PFM,A3M,A4PFM],{i,Length[I1]}],"RKPF","value of RKPF function"];
xplotz[GAMMA=Table[GAM[j3type,RK[[i]],Min[Max[1/180 LODE[[i]] \[Pi],-(\[Pi]/6)],\[Pi]/6]],{i,1,Length[LODE]}],"GAMMA","J3type \[CapitalGamma] function"];
xplotz[lode=(LODE \[Pi])/180,"lode","Lode angle in radians"];xplotz[GAMMAopposite=Table[GAM[j3type,1/3 ArcSin[Sin[3/180 (LODE[[i]]+180) \[Pi]]]],{i,1,Length[LODE]}],"GAMMAopposite","value of J3Type \[CapitalGamma] function on opposite side of YS"];
xplotz[LodeXsig=LodeX[sig11,sig22,sig33],"LodeXsig","Lode Cartesian x-coordinate"];
xplotz[LodeYsig=LodeY[sig11,sig22,sig33],"LodeYsig","Lode Cartesian y-coordinate"];
xplotz[LodeXbs=LodeX[QSBSXX,QSBSYY,QSBSZZ],"LodeXbs","Lode Cartesian x-coordinate for backstress"];xplotz[LodeYbs=LodeY[QSBSXX,QSBSYY,QSBSZZ],"LodeYbs","Lode Cartesian y-coordinate for backstress"];xplotz[trueLodesig=TrueLode[LodeXsig,LodeYsig],"trueLodesig","wide range Lode Angle"];
xplotz[porosity=1-E^(-EQPV-P3U),"porosity","volume fraction of pores"];
xplotz[computedX=Table[QSEL[[i]]-CRM (A1M-A3M E^(A2M QSEL[[i]])-A4M QSEL[[i]]),{i,Length[QSEL]}],"computedX","X branch point (should equal KAPPA)"];

xplotz[computedEVP=Table[P3M (E^(P1M (computedX[[i]]-P0M)-P2M (computedX[[i]]-P0M)^2)-1),{i,Length[computedX]}],"computedEVP","computed plastic volumetric strain"];
xpropz[uniStiffU=B0U+(4/3)G0U,"uniStiffU"];
xpropz[uniStiffM=B0M+(4/3)G0M,"uniStiffM"];


meridpath=ListPlot[Transpose[{sigm,sigs}],Joined->True,PlotStyle->{myStyle},PlotRange->sigsrange];]
init["computedVARS","evaluates GeoModel variables that might be useful in calculations","computedVARS[]"];
init["geocrack","True if the calculation includes damage","geocrack"];
init["pres","pressure determined in Mathemcatica from stress components","pres"];
init["PRES","pressure reported in simulation output","PRES"];
init["vstrain","volumetric strain computed in Mathematic from trace of strain ","vstrain"];
init["EVOL","volumetric strain computed within the simulation ","EVOL"];
init["sigm","z Lode coordinate (isomorphic mean stress)","sigm"];
init["sigd<ij>","components of the stress deviator","sigd11,22,33"];
init["sigs","r Lode coordinate (isomorphic shear stress)","sigs"];
init["sigsrange","range of r-Lode coordinate during the simulation",sigsrange];
init["scaleRJ2","a characteristic scale for ROOTJ2 (to normalize plots)",scaleRJ2];
init["scalez","characteristic scale for z-Lode coordinate",scalez];
init["j3type","integer value of J3TYPE",j3type];
init["GAMMA","pressure varying value of the GAMMA function","GAMMA"];
init["lode","Lode angle in radians","lode"];
init["GAMMAopposite","value of the GAMMA function on the opposite side of yield surface","GAMMAopposite"];
init["LodeXsig","true x-Cartesian-Lode coordinate","LodeXsig"];
init["LodeYsig","true y-Cartesian-Lode coordinate","LodeYsig"];
init["LodeXbs","same as LodeXsig, but for backstress","LodeXbs"];
init["LodeYbs","same as LodeYsig, but for backstress","LodeYbs"];
init["trueLodesig","true Lode angle","trueLodesig"];
init["porosity","estimate for porosity based on volumetric plastic strain","porosity"];
init["RK","pressure varying value of the strength ratio","RK"];
init["RKPF","pressure varying strength ratio for the nonassociative potential","RKPF"];
init["computedX","value of I1 at the cap","computedX"];
init["computedEVP","Mathematica computed EVP (should equal code's EVP)","computedEVP"];
init["meridpath","plot of sigs vs. sigm","meridpath"];


meridianALLold[nmerid_,scale_]:=Module[{},rmin=sigsrange[[1]];rmax=sigsrange[[2]];zbarMin=Min[-I1]/Sqrt[3];zbarMax=Max[-I1]/Sqrt[3];If[zbarMax-zbarMin<B0M/10^9,zbarMax=1;zbarMin=-1];dm=IntegerPart[(lastep-1)/Min[nmerid,lastep-1]];mstep=Table[kkk,{kkk,1,lastep,dm}];meridTABLE=Table[1/scale merid[scale zbar,A1M,A2M,A3M,A4M,RNM GFUN[[mstep[[i]]]],-KAPPA[[mstep[[i]]]],CRM],{i,1,Length[mstep]}];meridFam=Plot[Evaluate[meridTABLE],{zbar,zbarMin/scale,zbarMax/scale},PlotStyle->{green},PlotRange->{rmin/scale,(rmax 1.01`)/scale},PlotPoints->100];limitcurve=Plot[limitCurve[scale zbar,A1M,A2M,A3M,A4M]/scale,{zbar,zbarMin/scale,zbarMax/scale},PlotRange->{rmin/scale,(rmax 1.01`)/scale},PlotStyle->{blue},PlotPoints->100];meridpathscaled=ListPlot[Transpose[{sigm/scale,sigs/scale}],Joined->True,PlotRange->sigsrange/scale];Show[meridFam,limitcurve,meridpathscaled,ImageSize->72 8]]


meridianALLbasicNOFLOW[nmerid_,scale_]:=Module[{},rmin=sigsrange[[1]];rmax=sigsrange[[2]];zbarMin=Min[-I1]/Sqrt[3];zbarMax=Max[-I1]/Sqrt[3];If[zbarMax-zbarMin<B0M/10^9,zbarMax=1;zbarMin=-1];dm=IntegerPart[(lastep-1)/Min[nmerid,lastep-1]];mstep=Table[kkk,{kkk,1,lastep,dm}];meridTABLE=Table[1/scale merid[scale zbar,A1M,A2M,A3M,A4M,RNM GFUN[[mstep[[i]]]],-KAPPA[[mstep[[i]]]],CRM],{i,1,Length[mstep]}];meridFam=Plot[Evaluate[meridTABLE],{zbar,zbarMin/scale,zbarMax/scale},PlotStyle->{green},PlotRange->{rmin/scale,(rmax 1.01`)/scale},PlotPoints->100];limitcurve=Plot[limitCurve[scale zbar,A1M,A2M,A3M,A4M]/scale,{zbar,zbarMin/scale,zbarMax/scale},PlotRange->{rmin/scale,(rmax 1.01`)/scale},PlotStyle->{blue},PlotPoints->100];meridpathscaled=ListPlot[Transpose[{sigm/scale,sigs/scale}],Joined->True,PlotRange->sigsrange/scale];Show[meridFam,limitcurve,meridpathscaled,ImageSize->72 8]]


meridianALLbasic[nmerid_,scale_]:=Module[{},rmin=Max[sigsrange[[1]] Max[GAMMA],0];rmax=sigsrange[[2]] Max[GAMMA];Print["here i am 0"];
zbarMin=Min[-I1]/Sqrt[3];zbarMax=Max[-I1]/Sqrt[3];If[zbarMax-zbarMin<B0M/10^9,zbarMax=1;zbarMin=-1];dm=IntegerPart[(lastep-1)/Min[nmerid,lastep-1]];mstep=Table[kkk,{kkk,1,lastep,dm}];meridTABLE=Table[1/scale merid[scale zbar,A1M,A2M,A3M,A4M,RNM GFUN[[mstep[[i]]]],-KAPPA[[mstep[[i]]]],CRM],{i,1,Length[mstep]}];
Print["here i am 1"];
meridFam=Plot[Evaluate[meridTABLE],{zbar,zbarMin/scale,zbarMax/scale},PlotStyle->{green},PlotRange->{rmin/scale,(rmax 1.01`)/scale},PlotPoints->100];limitcurve=Plot[limitCurve[scale zbar,A1M,A2M,A3M,A4M]/scale,{zbar,zbarMin/scale,zbarMax/scale},PlotRange->{rmin/scale,(rmax 1.01`)/scale},PlotStyle->{RGBColor[0.6,0.6,1],Thickness[0.005]},PlotPoints->100];limitcurvePF=Plot[limitCurve[scale zbar,A1M,A2PFM,A3M,A4PFM]/scale,{zbar,zbarMin/scale,zbarMax/scale},PlotRange->{rmin/scale,(rmax 1.05)/scale},PlotStyle->{red,Thickness[0.003],Dashed},PlotPoints->100];
prinlineTXC=Plot[Sqrt[2] z+(Sqrt[6] Min[CTPSM,10.0^10])/scale,{z,zbarMin/scale,zbarMax/scale},PlotRange->{rmin/scale,(rmax 1.01)/scale},PlotStyle->{Hue[0.5],chainDash,Thick}];
prinlineTXE=Plot[1/2 (Sqrt[2] z+(Sqrt[6] Min[CTPSM,10.0^10])/scale),{z,zbarMin/scale,zbarMax/scale},PlotRange->{rmin/scale,(rmax 1.01)/scale},PlotStyle->{Hue[0.5],chainDash,Thick}];Print["cyan chaindash=prinline"];
meridpathscaled=ListPlot[Transpose[{sigm/scale,(sigs GAMMA)/scale}],Joined->True,PlotRange->sigsrange/scale,PlotStyle->{Thickness[0.005`]}];Show[prinlineTXC,prinlineTXE,meridFam,limitcurve,limitcurvePF,meridpathscaled,ImageSize->72 8]]
init["meridianALLbasic","function that creates a basic view of the meridional path","meridianALLbasic[number_of_profiles,scale]"];


yval[r_,theta_]:=2r Sin[theta]


meridianALL[nmerid_,scale_]:=Module[{gamSHR,prng,limitCurveTXCcolor=RGBColor[0,.8,0],limitCurveTXEcolor=Hue[0.1],prinLinecolor=Hue[0.5],meridFamColor=green},
rmin=sigsrange[[1]];rmax=sigsrange[[2]];
zbarMin=Min[-I1]/Sqrt[3];zbarMax=Max[-I1]/Sqrt[3];
prng={Min[(rmin 1.2)/scale,-((rmax 1.2)/scale)],Max[(rmax 1.2)/scale,-((rmin 1.2)/scale)]};
If[prng[[2]]-prng[[1]]<(zbarMax-zbarMin)/scale,prng={Mean[prng]-(zbarMax-zbarMin)/(2scale),Mean[prng]+(zbarMax-zbarMin)/(2scale)}];
If[zbarMax-zbarMin<B0M/10^9,zbarMax=1;zbarMin=-1];dm=IntegerPart[(lastep-1)/Min[nmerid,lastep-1]];mstep=Table[kkk,{kkk,2,lastep,dm}];mstep=Join[mstep,{lastep}];meridTABLE=Flatten[Table[{yval[merid[scale zbar,A1M,A2M,A3M,A4M,RNM GFUN[[mstep[[i]]]],-KAPPA[[mstep[[i]]]],CRM]/(scale GAMMA[[mstep[[i]]]]),lode[[mstep[[i]]]]],-yval[merid[scale zbar,A1M,A2M,A3M,A4M,RNM GFUN[[mstep[[i]]]],-KAPPA[[mstep[[i]]]],CRM]/(scale GAMMAopposite[[mstep[[i]]]]),lode[[mstep[[i]]]]]},{i,1,Length[mstep]}]];
meridFam=Plot[Evaluate[meridTABLE],{zbar,zbarMin/scale,zbarMax/scale},PlotStyle->{meridFamColor},PlotRange->{-((rmax 1.01)/scale),(rmax 1.01)/scale},PlotPoints->100];
limitcurve=Plot[limitCurve[scale zbar,A1M,A2M,A3M,A4M]/scale,{zbar,zbarMin/scale,zbarMax/scale},PlotRange->prng,PlotStyle->{limitCurveTXCcolor,Dashing[{0.02,0.01}]},PlotPoints->100];gamSHR=GAM[j3type,0];
prinlineTXC=Plot[Sqrt[2] z+(Sqrt[6] Min[CTPSM,10.0^10])/scale,{z,zbarMin/scale,zbarMax/scale},PlotRange->prng,PlotStyle->{prinLinecolor,chainDash}];
prinlineTXE=Plot[-1/2 (Sqrt[2] z+(Sqrt[6] Min[CTPSM,10.0^10])/scale),{z,zbarMin/scale,zbarMax/scale},PlotRange->prng,PlotStyle->{prinLinecolor,chainDash}];
limitcurveBot=Plot[-(limitCurveTXE[RK0,scale zbar,A1M,A2M,A3M,A4M] /scale),{zbar,zbarMin/scale,zbarMax/scale},PlotRange->prng,PlotStyle->{limitCurveTXEcolor,Dashing[{0.02,0.01}]},PlotPoints->100];meridpathscaled=ListPlot[Table[{sigm[[i]]/scale,yval[sigs[[i]],lode[[i]]]/scale},{i,1,lastep}],Joined->True,PlotRange->sigsrange/scale,PlotStyle->{myThickness}];meridtest=Show[meridFam,limitcurve,limitcurveBot,prinlineTXC,prinlineTXE,meridpathscaled,PlotRange->prng,ImageSize->72 8];
Print[
Style["  dash=limitCurveTXE",FontColor->limitCurveTXEcolor],
Style["  solid=meridFam",FontColor->meridFamColor],
Style["  dash=limitCurveTXC",FontColor->limitCurveTXCcolor],
Style["  chaindash=prinLine",FontColor->prinLinecolor]
];
If[MakeGraphs,Export[demodir<>"meridPath.gif",Show[meridtest]],Show[meridtest,PlotRange->prng]]
]
init["meridianALL","function that creates a rotated side view of the meridional path","meridianALL"];


meridianALL[nmerid_,zbarLO_,zbarHI_,rLO_,rHI_]:=(dm=IntegerPart[(lastep-1)/Min[Max[nmerid,1],lastep-1]];mstep=Table[kkk,{kkk,1,lastep,dm}];meridTABLE=Table[merid[zbar,A1M,A2M,A3M,A4M,RNM GFUN[[mstep[[i]]]],-KAPPA[[mstep[[i]]]],CRM],{i,1,Length[mstep]}];meridFam=Plot[Evaluate[meridTABLE],{zbar,zbarLO,zbarHI},PlotStyle->{green},PlotRange->{rLO,rHI},PlotPoints->100];Show[meridFam,meridpath])


elasticInfo=.;ClearAll[elasticInfo];
elasticInfo[]:=Module[{RAT,facN,facS,p11,p12,p21,p22,barg,garg,bkmoda,shmoda},RAT=G0M/B0M;facN=(RJSM RKNM)/B0M;facS=(RJSM RKSM)/G0M;
xpropz[PoisU=(3 B0U-2 G0U)/2/(3 B0U + G0U),"PoisU"];
xpropz[PoisM = (3 B0M-2 G0M)/2/(3 B0M + G0M),"PoisM"];
Print[TableForm[{{"bulk modulus parameters =",B0M,B1M,B2M,B3M,B4M},{"shear modulus parameters =",G0M,G1M,G2M,G3M,G4M},{"joint degredation of bulk modulus, facK=",facK=If[RJSM>0,(.75*facN+RAT)/(.75*(1+facN)+RAT),1]},{"joint degredation of shear modulus, facG=",facG=If[RJSM>0,(.6+facS)/(1+facS)+(4./15)*(facK-1)*RAT,1]}}]];absI1=Table[Max[-(QSSIGXX[[i]]+QSSIGYY[[i]]+QSSIGZZ[[i]]),1/1000000000000000000000000000000000000000],{i,1,Length[I1]}];rootJ2qs=\[Sqrt](1/3 (QSSIGXX^2+QSSIGYY^2+QSSIGZZ^2-QSSIGXX QSSIGYY-QSSIGYY QSSIGZZ-QSSIGZZ QSSIGXX)+QSSIGXY^2+QSSIGYZ^2+QSSIGZX^2);barg=Table[-(B4M/Max[Abs[EVP[Xbar[-QSEL[[i]],CRM,A1M,A2M,A3M,A4M]]],1/10^60]),{i,1,Length[EQPV]}];garg=Table[-(G4M/Max[Abs[EQPS[[i]]],1/10^60]),{i,1,Length[EQPS]}];bkmoda=facK (B0M+B1M Exps[-(B2M/absI1)]);shmoda=(facG (G0M (1-G1M E^(-G2M rootJ2qs))))/(1-G1M);bkmodA=facK (B0M+B1M Exps[-(B2M/absI1)]-B3M Exps[barg]);shmodA=facG ((G0M (1-G1M E^(-G2M rootJ2qs)))/(1-G1M)-G3M Exps[garg]);poisA=(3 bkmodA-2 shmodA)/(2 (3 bkmodA+shmodA));usmA=bkmodA+(4 shmodA)/3;p11=Show[ListPlot[Table[{time[[i]],bkmodA[[i]]/B0M},{i,1,lastep}],Joined->True,AxesLabel->{"time","bkmodA/B0M"},PlotStyle->{fatLine}],ListPlot[Table[{time[[i]],bkmoda[[i]]/B0M},{i,1,lastep}],Joined->True,AxesLabel->{"time","bkmoda/B0M"},PlotStyle->{thinLine,green}]];p12=Show[ListPlot[Table[{time[[i]],shmodA[[i]]/G0M},{i,1,lastep}],Joined->True,AxesLabel->{"time","shmodA/G0M"},PlotStyle->{fatLine}],ListPlot[Table[{time[[i]],shmoda[[i]]/G0M},{i,1,lastep}],Joined->True,AxesLabel->{"time","shmoda/G0M"},PlotStyle->{thinLine,green}]];p21=Show[ListPlot[Table[{time[[i]],USM[[i]]/(B0M+(4 G0M)/3)},{i,1,lastep}],Joined->True,AxesLabel->{"time","USM/USM_initial"},PlotStyle->{fatLine}],ListPlot[Table[{time[[i]],usmA[[i]]/(B0M+(4 G0M)/3)},{i,1,lastep}],Joined->True,AxesLabel->{"time","usmA/USM_initial"},PlotStyle->{red}]];p22=ListPlot[Table[{time[[i]],poisA[[i]]},{i,1,lastep}],Joined->True,AxesLabel->{"time","poisA"},PlotStyle->{blue}];Show[GraphicsGrid[{{p11,p12},{p21,p22}}],ImageSize->72 8]];
init["elasticInfo","function that computes elastic constants and other things elastic","elasticInfo[]"];
init["bkmodA","nonlinear bulk modulus","{bkmodA}"];
init["shmodA","nonlinear shear modulus","{shmodA}"];
init["poisA","nonlinear poisson ratio","{poisA}"];
init["usmA","computed nonlinear constrained modulus = K+4G/3","{usmA}"];
init["USM","constrained modulus output from code (should equal usmA)","{USM}"];
constrainedModulus:=B0M+(4 G0M)/3
init["constrainedModulus","initial constrained modulus","USM"];


ExactStyle={Thickness[0.02],RGBColor[1,.5,0]};


ModelStyle={Thickness[0.008],Dashing[{0.05,0.01}]};


ErrorStyle={RGBColor[.7,.7,.85],PointSize[0.03]};


mdlErrorTab[EXACTabscissa_,EXACTordinate_,MODELabscissa_,MODELordinate_]:=Module[{scl,exactordinate,npnts},scl=1/(Max[EXACTordinate]-Min[EXACTordinate]);exactordinate=interp[EXACTabscissa,EXACTordinate];npnts=Length[MODELordinate];
Table[Abs[(MODELordinate[[i]]-exactordinate[MODELabscissa[[i]]])scl],{i,1,npnts}]]
init["mdlErrorTab","function to compute model error table","mdlErrorTab[...]"];


MODELerror=.;Clear[MODELerror];
MODELerror[EXACTabscissa_,EXACTordinate_,MODELabscissa_,MODELordinate_]:=Module[{jnk,npnts},
jnk=mdlErrorTab[EXACTabscissa,EXACTordinate,MODELabscissa,MODELordinate];
npnts=Length[jnk];Log[10,Sqrt[Sum[jnk[[i]]^2,{i,1,npnts}]]/npnts]]
init["MODELerror","function that returns overall model error","MODELerror[...]"];


showMODELerror[EXACTabscissa_,EXACTordinate_,MODELabscissa_,MODELordinate_,scaleLO_,scaleHI_]:=Module[{jnk,ermax,eta},jnk=mdlErrorTab[EXACTabscissa,EXACTordinate,MODELabscissa,MODELordinate];ermax=Max[jnk];eta=jnk/ermax;ListPlot[scaleLO+(scaleHI-scaleLO) eta,PlotRange->All,PlotStyle->ErrorStyle]]
init["showMODELerror","function generating a plot of model error","showMODELerror[...]"];


showMODELerror[EXACTabscissa_,EXACTordinate_,MODELabscissa_,MODELordinate_]:=Module[{jnk,ermax,scaledErr,npnts},jnk=mdlErrorTab[EXACTabscissa,EXACTordinate,MODELabscissa,MODELordinate];
ermax=Max[jnk];npnts=Length[MODELordinate];scLO=Min[MODELordinate];scHI=Max[MODELordinate];scaledErr=Table[{MODELabscissa[[i]],scLO+(scHI-scLO)*jnk[[i]]/ermax},{i,npnts}];ListPlot[scaledErr,PlotRange->All,PlotStyle->ErrorStyle]]


showMODELerrori[EXACTabscissa_,EXACTordinate_,MODELabscissa_,MODELordinate_]:=Module[{jnk,ermax,scaledErr,npnts},jnk=mdlErrorTab[EXACTabscissa,EXACTordinate,MODELabscissa,MODELordinate];
ermax=Max[jnk];npnts=Length[MODELordinate];scLO=Min[MODELordinate];scHI=Max[MODELordinate];scaledErr=Table[{MODELabscissa[[i]],scLO+(scHI-scLO)*jnk[[i]]/ermax},{i,npnts}];ListPlot[scaledErr,PlotRange->All,PlotStyle->ErrorStyle]]
init["showMODELerrori","creates but does not display the model error plot","showMODELerrori[...]"];


compare[EXACTabscissa_,EXACTordinate_,MODELabscissa_,MODELordinate_]:=Show[makeme[EXACTabscissa,EXACTordinate,ExactStyle],makeme[MODELabscissa,MODELordinate,ModelStyle],ImageSize->2 4 72]
init["compare","gives a plot of computed and exact results","compare[...]"];


comparei[EXACTabscissa_,EXACTordinate_,MODELabscissa_,MODELordinate_]:=Show[makeme[EXACTabscissa,EXACTordinate,ExactStyle],makeme[MODELabscissa,MODELordinate,ModelStyle],ImageSize->2 4 72]
init["comparei","creates but does not display the model comparison plot","comparei[...]"];


compare[EXACTabscissa_,EXACTordinate_,MODELabscissa_,MODELordinate_,label_]:=Show[makeme[EXACTabscissa,EXACTordinate,ExactStyle],makeme[MODELabscissa,MODELordinate,ModelStyle],ImageSize->2 4 72,PlotLabel->label<>" log10error: "<>ToString[MODELerror[EXACTabscissa,EXACTordinate,MODELabscissa,MODELordinate]]]


comparei[EXACTabscissa_,EXACTordinate_,MODELabscissa_,MODELordinate_,label_]:=Show[makeme[EXACTabscissa,EXACTordinate,ExactStyle],makeme[MODELabscissa,MODELordinate,ModelStyle],ImageSize->2 4 72,PlotLabel->label<>" log10error: "<>ToString[MODELerror[EXACTabscissa,EXACTordinate,MODELabscissa,MODELordinate]]]


cmpr[EXACTabscissa_,EXACTordinate_,MODELabscissa_,MODELordinate_]:=Show[comparei[EXACTabscissa,EXACTordinate,MODELabscissa,MODELordinate],showMODELerrori[EXACTabscissa,EXACTordinate,MODELabscissa,MODELordinate],ImageSize->4 72]
init["cmpr","overlays a comparison plot with a scaled error plot","cmpr[...]"];


cmpr[EXACTabscissa_,EXACTordinate_,MODELabscissa_,MODELordinate_,label_]:=Show[comparei[EXACTabscissa,EXACTordinate,MODELabscissa,MODELordinate,label],showMODELerrori[EXACTabscissa,EXACTordinate,MODELabscissa,MODELordinate],ImageSize->4 72]


seti1i2[i1try_,i2try_]:=(If[i1try>lastep,i1=1,i1=i1try];If[i2try>lastep,i2=lastep,i2=i2try];)


init["seti1i2","sets values to temporary i1 and i2","seti1i2[i1value,i2value]"];


Sort[initlist]
