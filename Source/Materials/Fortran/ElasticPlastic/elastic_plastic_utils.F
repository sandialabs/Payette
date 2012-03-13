C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMTD(A,AISO,ADEV,FINVA,SINVA)
C***********************************************************************
C     PURPOSE: Return deviatoric part of second order tensor
C
C input
C -----
C    A: Second order tensor
C
C output
C -----
C    TRA:  Trace of A
C    AISO: Isotropic part of A
C    ADEV: Deviatoric part of A
C
C  MODIFICATION HISTORY
C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100223|tjfulle|created routine
C
#ifdef TJF_IMPLNONE
      IMPLICIT NONE
#else
#include "precision.Blk"
#endif
C............................................................parameters
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION PZERO,PONE,PTHREE,PTHIRD,PHALF
#endif
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTHREE=3.D0)
      PARAMETER (PTHIRD=0.333333333333333333333333333333333333333333D0)
      PARAMETER (PHALF=0.5D0)
C.................................................................passed
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION A,FINVA,SINVA,AISO,ADEV
#endif
      DIMENSION A(6),AISO(6),ADEV(6)
C..................................................................local
#ifdef TJF_IMPLNONE
      INTEGER IJ
      DOUBLE PRECISION TMP,DELTA
#endif
      DIMENSION DELTA(6)
      SAVE DELTA
      DATA DELTA/PONE,PONE,PONE,PZERO,PZERO,PZERO/
C...............................................................external
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION DMMDBD
#endif
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      FINVA= A(1)+A(2)+A(3)
      DO IJ=1,6
         AISO(IJ)= PTHIRD*FINVA*DELTA(IJ)
         ADEV(IJ)= A(IJ) - AISO(IJ)
      ENDDO
C     Check that deviatoric
      TMP= (ADEV(1)+ADEV(2)+ADEV(3))*PTHIRD
      IF(TMP.NE.PZERO)THEN
         ADEV(1)=ADEV(1)-TMP
         ADEV(2)=ADEV(2)-TMP
         ADEV(3)=-(ADEV(1)+ADEV(2))
      ENDIF
      SINVA= PHALF*DMMDBD(ADEV,ADEV)
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMDEV(A,ADEV)
C***********************************************************************
C     PURPOSE: Return deviatoric part of second order tensor
#ifdef TJF_IMPLNONE
      IMPLICIT NONE
#else
#include "precision.Blk"
#endif
C............................................................parameters
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION PZERO,PONE,PTHREE,PTHIRD
#endif
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTHREE=3.D0,PTHIRD=PONE/PTHREE)
C.................................................................passed
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION A,ADEV
#endif
      DIMENSION A(6),ADEV(6)
C..................................................................local
#ifdef TJF_IMPLNONE
      INTEGER IJ
      DOUBLE PRECISION TMP,DELTA
#endif
      DIMENSION DELTA(6)
      SAVE DELTA
      DATA DELTA/PONE,PONE,PONE,PZERO,PZERO,PZERO/
C...............................................................external
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      TMP= PTHIRD*(A(1)+A(2)+A(3))
      DO IJ=1,6
         ADEV(IJ)= A(IJ) - TMP*DELTA(IJ)
      ENDDO
C     Check that deviatoric
      TMP= (ADEV(1)+ADEV(2)+ADEV(3))*PTHIRD
      IF(TMP.NE.PZERO)THEN
         ADEV(1)=ADEV(1)-TMP
         ADEV(2)=ADEV(2)-TMP
         ADEV(3)=-(ADEV(1)+ADEV(2))
      ENDIF
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMPAR(UI)
C***********************************************************************
C     PURPOSE: This routine transfers property values in the "UI" array
C     into the DMMPROP common blocks and computes derived constants
C
C input
C -----
C    UI: property array
C
C output
C -----
C    /DMMPROPL/
C    /DMMPROPI/
C    /DMMPROPR/
C
C  MODIFICATION HISTORY
C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100223|tjfulle|created routine
C
#ifdef TJF_IMPLNONE
      IMPLICIT NONE
#else
#include "precision.Blk"
#endif
C.............................................................parameters
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION PZERO,PTWO,PTHREE,PSIX,ROOT3,TOOR3,ROOT23,PTHIRD
#endif
      PARAMETER (PZERO=0.0D0,PTWO=0.2D1,PTHREE=0.3D1,PSIX=6.0D0)
      PARAMETER (PTHIRD=0.3333333333333333333333333333333333333333333D0)
      PARAMETER (  ROOT3=0.173205080756887729352744634150587236694281D1)
      PARAMETER (  TOOR3=0.577350269189625764509148780501957455647602D0)
      PARAMETER ( ROOT23=0.816496580927726032732428024901963797321982D0)
#include "elastic_plastic_pnt.Blk"
C.................................................................common
#include "elastic_plastic_prop.h"
C.................................................................passed
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION UI
#endif
      DIMENSION UI(*)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC DMMPAR
C     Logicals
      ANISO=(NINT(UI(IPAN)).NE.0)
      RATEDEP=(UI(IPT1).NE.PZERO)
      WANTSELASTIC=(UI(IPA1).GT.1D90)
C     Properties
      B0= UI(IPB0)
      B1= UI(IPB1)
      B2= UI(IPB2)
      G0= UI(IPG0)
      IF(ANISO)THEN
         G1= PTHIRD*UI(IPG1)
         TWOG1=PTWO*G1
      ELSE
         G1= UI(IPG1)
         TWOG1=PZERO
      ENDIF
      G2= UI(IPG2)
      G3= UI(IPG3)
      PRAT= (PTHREE*B0-PTWO*G0)/(PSIX*B0+PTWO*G0)
      A1= TOOR3*UI(IPA1)
      A2= UI(IPA2)
      A3= UI(IPA3)
      A4= UI(IPA4)
      A4G= UI(IPA4PF)
      PRESDEPYLD=(A3.GT.PZERO.OR.A4G.GT.PZERO)
      RHO0= UI(IPRHO0)
      T0= UI(IPT0)
      TM= UI(IPTM)
      XP= UI(IPXP)
      C0= UI(IPC0)
      CV= UI(IPCV)
      GP= UI(IPGP)
      S1= UI(IPS1)
      T1= UI(IPT1)
      T2= UI(IPT2)
      T3= UI(IPT3)
      T4= UI(IPT4)
      RETURN
      END ! SUBROUTINE DMMPAR

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMLOD(A,AZ,AR,ER)
C***********************************************************************
C     PURPOSE: Return Lode components of second order tensor A
C
C input
C -----
C    A: Second order tensor
C
C output
C -----
C    AZ:  Z component of A
C    AR:  S component of A
C    ER:  Basis tensor of deviatoric part of A
C

C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100223|tjfulle|created routine
C
#ifdef TJF_IMPLNONE
      IMPLICIT NONE
#else
#include "precision.Blk"
#endif
C.............................................................parameters
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION TOL1M10,TOOR3,HUGE
#endif
      PARAMETER (TOL1M10=1.D-10,HUGE=1.D90)
      PARAMETER (  TOOR3=0.577350269189625764509148780501957455647602D0)
C.................................................................common
C.................................................................passed
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION A,AZ,AR,ER
#endif
      DIMENSION A(6),ER(6)
C...............................................................external
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION DMMTR,DMMTMG
#endif
C..........................................................local scalars
#ifdef TJF_IMPLNONE
      INTEGER IJ
      DOUBLE PRECISION DUM
#endif
C...........................................................local arrays
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION S
#endif
      DIMENSION S(6)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC DMMLOD
      AZ = TOOR3*DMMTR(A)
      CALL DMMDEV(A,S)
      AR = DMMTMG(S)
      DUM=AR
      IF(AR.LT.TOL1M10)DUM=HUGE
      DO IJ=1,6
         ER(IJ) = S(IJ)/DUM
      ENDDO
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMMOD(UI,U,RHO,T,EV,TAU,BM,SM)
C***********************************************************************
C     PURPOSE: Determines non-linear elastic properties
C
C input
C -----
C    UI: UI array
C    U:  Internal energy
C    RHO:Density
C    T:  Temperature
C    EV: Elastic volume strain
C    RI1:   First invariant of stress tensor
C
C output
C -----
C    BM:  Bulk modulus
C    SM:  Shear modulus
C
C  MODIFICATION HISTORY
C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100223|tjfulle|created routine
C
#ifdef TJF_IMPLNONE
      IMPLICIT NONE
#else
#include "precision.Blk"
#endif
C............................................................parameters
#include "elastic_plastic_pnt.Blk"
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION PZERO,PHALF,PONE,PTWO,PTHREE,PTEN,PTHIRD
#endif
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTWO=2.D0,PTHREE=3.D0,PTEN=10.D0)
      PARAMETER (PHALF=0.5D0)
      PARAMETER (PTHIRD=0.333333333333333333333333333333333333333333D0)
C.................................................................common
#include "elastic_plastic_prop.h"
C.................................................................passed
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION UI,U,RHO,T,EV,BM,SM,TAU
#endif
      DIMENSION UI(*),TAU(6)
C..................................................................local
#ifdef TJF_IMPLNONE
      INTEGER IDK,IDG
      DOUBLE PRECISION P,S,UIMG
      DOUBLE PRECISION CS,RJFAC
#endif
#ifdef KERLEY_EOS_RTNS
      DIMENSION S(9),UIMG(22)
#endif
C...................................................................data
C...............................................................external
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION DMMTR
#endif
C....................................................statement functions
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc DMMMOD
      IDK=UI(IPIDK)
      IDG=UI(IPIDG)
      P= -PTHIRD*DMMTR(TAU)
      CS = PONE
      U = PONE

C     Compute bulk modulus
      IF(IDK.EQ.0)THEN
         IF(EV.LT.PZERO)THEN
            CS= SQRT((B0 + B1*EV + PHALF*B2*EV**2)/RHO0)
         ELSE
            CS= SQRT(B0/RHO0)
         ENDIF

#ifdef KERLEY_EOS_RTNS
      ELSEIF(IDK.EQ.1)THEN
C     Kerley EOS to compute sound speed
         CALL EOSUI(UI,UIMG)
         CALL EOSMGV(1,1,UIMG,UI,UI(IPDC1),RHO,U,0,0,P,T,CS,S)
#endif
      ENDIF
      BM= RHO0*CS**2

C     Compute shear modulus
      IF(IDG.EQ.0)THEN
         SM= MAX(G0,G0 + G1*EV + G2*(T-300.D0))

      ELSEIF(IDG.EQ.1)THEN
C     Shear modulus from SGC formula with elastic-plastic coupling
         RJFAC= (RHO0/RHO)**PTHIRD
         SM= MAX(G0,G0 + G1*(EV*RJFAC) + G2*(T-300.D0))

      ELSEIF(IDG.EQ.2)THEN
C     Shear modulus from elasticity relations
         SM= PTHREE*BM*(PONE-PTWO*PRAT)/(PTWO*(PONE+PRAT))
      ENDIF

C     Set shear modulus to zero if melt temp is exceeded
      IF(T.GE.TM)THEN
         CALL LOGMES('Melt temp reached, setting G=0')
         SM= G0
      ENDIF

      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE OVERSTRESS(DT,SV,EQDOT,SIGOLD,DSIGT,QSSIGOLD,QSSIG,SIG)
C***********************************************************************
C     PURPOSE: Compute Duvaut-Lions overstress
C
C input
C -----
C    DT:       Timestep
C    SV:       State variable array
C    EQDOT:    Equivalent strain rate
C    SIGOLD:   Stress at beginning of timestep
C    DSIGT:    Trial stress increment
C    QSSIGOLD: Quasistatic stress at beginning of timestep
C    QSSIG:    Current quasistatic stress
C
C output
C -----
C    SIG: Dynamic stress at end of step
C
C  MODIFICATION HISTORY
C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100223|tjfulle|ported/modified Kayenta routine for DMM
C
#ifdef TJF_IMPLNONE
      IMPLICIT NONE
#else
#include "precision.Blk"
#endif
C
C.............................................................parameters
#include "elastic_plastic_pnt.Blk"
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION PTHIRD,PHALF,PZERO,PONE,PSIX,TOL1M20,EUNDERFLOW
#endif
      PARAMETER (PTHIRD=0.3333333333333333333333333333333333333333333D0)
      PARAMETER (PHALF=0.5D0)
      PARAMETER (PZERO=0.D0,PONE=1.D0,PSIX=6.D0)
      PARAMETER (TOL1M20=1.D-20)
      PARAMETER (EUNDERFLOW=-34.53877639491D0*PONE)
C.................................................................common
#include "elastic_plastic_prop.h"
C.................................................................passed
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION DT,EQDOT,SIGOLD,QSSIGOLD,QSSIG,DSIGT,SIG,SV
#endif
      DIMENSION SIGOLD(6),QSSIGOLD(6),QSSIG(6),DSIGT(6),SIG(6),SV(*)
C..................................................................local
#ifdef TJF_IMPLNONE
      INTEGER IJ,I
      DOUBLE PRECISION SIGV,DUM,TCHAR,RHI,RLO,RMID,RAT
#endif
      DIMENSION SIGV(6)
C...................................................................data
C...............................................................external
C....................................................statement functions
C
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc OVERSTRESS
      IF(.NOT.RATEDEP)THEN
         DO IJ=1,6
            SIG(IJ)= QSSIG(IJ)
         ENDDO
         DO I=1,NDMMISV
            SV(I)= SV(I)
         ENDDO
      ENDIF
      GO TO 777
      call bombed('rate dependence has not been set up')

      dum=sv(1)
C     Set value for relaxation time. The value of tau is set based on
C     data being available for sig_over/sig_qs vs. log(eqdot) for
C     uniaxial strain. It is assumed that the data show a bilinear
C     relationship.


      !RNUM = abs(CHI*KMMDBD(SIG,D(1,NBLK),6))
      !DNOM = KMMDBD(PPPD,D(1,NBLK),6)*KMMDBD(QQQD,D(1,NBLK),6)
      !DUM=RNUM/DNOM
      IF(T2.EQ.PZERO)THEN
         TCHAR = T1
      ELSEIF(T3.EQ.PZERO.OR.EQDOT.LT.T3)THEN
         call bombed('Tim: dum is being used before initialized')
         TCHAR = DUM*(T1+T2*LOG(EQDOT)-PONE)
      ELSE
         call bombed('Tim: dum is being used before initialized')
         TCHAR = DUM*(T1+(T2-T4)*LOG(T3)+T4*LOG(EQDOT)-PONE)
      ENDIF

      IF(TCHAR.GT.TOL1M20)THEN
C     Manage underflow (which will result in overflow for RHI):
         RAT=DT/TCHAR
         IF(RAT.GT.ABS(EUNDERFLOW))THEN
C     ...extremely large timesteps (large value of rat)
C     (this usually also means low rates -- see comment below)
            RHI=PZERO
            RLO=PONE
            IF(RAT.LT.1.D40)THEN
               RMID=-PONE/RAT
            ELSE
               RMID=PZERO
            ENDIF
         ELSEIF(RAT.GT.1.D-3)THEN
C     ...intermediate timesteps
            RHI=(PONE-EXP(-RAT))/RAT
            RLO=PONE-RHI
            RMID=EXP(-RAT)-RHI
         ELSE
C     ...extremely small timesteps (small value of rat)
C     (this usually means high rates -- see comment below)
C     Use a series expansions of the above expressions
            RLO=RAT*(PHALF-RAT/PSIX)
            RHI=PONE-RLO
            RMID=RAT*(RAT*PTHIRD-PHALF)
         ENDIF
         DO IJ=1,6
            SIGV(IJ) = SIGOLD(IJ) + DSIGT(IJ)
            SIG(IJ) = RHI*SIGV(IJ) + RLO*QSSIG(IJ)
     &           + RMID*(SIGOLD(IJ) - QSSIGOLD(IJ))
         ENDDO
      ENDIF
 777  CONTINUE
      RETURN
      END


C***********************************************************************
C                       R M M  F U N C T I O N S
C***********************************************************************
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMTMG(A)
C     Purpose: Magnitude of second order tensor A
#include "precision.Blk"
      DIMENSION A(6)
      DMMTMG=SQRT(DMMDBD(A,A))
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMDBD(A,B)
C     Purpose: Double dot product of second order tensors A and B
#include "precision.Blk"
      PARAMETER (PTWO=2.D0)
      DIMENSION A(6),B(6)
      DMMDBD=     A(1)*B(1)+A(2)*B(2)+A(3)*B(3)
     $    + PTWO*(A(4)*B(4)+A(5)*B(5)+A(6)*B(6))
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMTR(A)
C     Purpose: Trace of second order tensor A
#include "precision.Blk"
      DIMENSION A(6)
      PARAMETER (TOL=1.D-15,PZERO=0.D0)
      DMMTR=A(1)+A(2)+A(3)
      IF(ABS(DMMTR).LT.TOL)DMMTR=PZERO
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMRJ2(A)
C     Purpose: Square root of second invariant of second order tensor A
#include "precision.Blk"
      DIMENSION A(6),AD(6)
      PARAMETER (PHALF=0.5D0)
      CALL DMMDEV(A,AD)
      DMMRJ2=SQRT(PHALF*DMMDBD(AD,AD))
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMEN(A,B,RHO)
C     Purpose: Energy inc. for stress A, strain inc. B and density RHO
#include "precision.Blk"
      DIMENSION A(6),B(6)
      DMMEN= DMMDBD(A,B)/RHO
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMTS(T)
C     Purpose: Johnson-Cook homologous temperature
#include "precision.Blk"
#include "elastic_plastic_prop.h"
      PARAMETER(PZERO=0.D0,PONE=1.D0)
      TARG= MAX(T-T0,PZERO)
      DMMTS= PONE-(TARG/(TM-T0))**XP
      dmmtmult= pone
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMYLD(TAU,EQPS,T)
C     Purpose: Value of yield function
#include "precision.Blk"
#include "elastic_plastic_prop.h"
      PARAMETER(ROOT23=0.816496580927726032732428024901963797321982D0)
      DIMENSION TAU(6)
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      RI1=DMMTR(TAU)
      FF= A1 - A3*EXP(A2*RI1) - A4*RI1 + A5*(ROOT23*EQPS)**A6
      DMMYLD= DMMRJ2(TAU) - FF*DMMTS(T)
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMDYLDFNCDT(D1,D2,D3)
C     Purpose: Derivative of yield function wrt T
#include "precision.Blk"
#include "elastic_plastic_prop.h"
      PARAMETER(PZERO=0.D0)
      TARG= MAX(D3-T0,PZERO)
      IF(TARG.EQ.PZERO)THEN
         DMMDYLDFNCDT= PZERO
      ELSE
         DMMDYLDFNCDT= (A1-A2*D1+A3*D2**A4)*XP/TARG*(TARG/(TM-T0))**XP
      ENDIF
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMHYFNC(A3,A4,EQPS,RMS)
C     Purpose:
#include "precision.Blk"
      PARAMETER(PZERO=0.D0)
      IF(EQPS.GT.PZERO)THEN
         HY= A3*A4*EQPS**A4*SQRT(RMS*RMS)/EQPS
      ELSE
         HY= PZERO
      ENDIF
ctim
      DMMHYFNC=HY
      dmmhyfnc=PZERO
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMHTFNC(TAUN,RM,RHO0,CV)
C     Purpose:
#include "precision.Blk"
      PARAMETER (PZERO=0.D0)
      DIMENSION TAUN(6),RM(6)
      DMMHTFNC= DMMEN(TAUN,RM,RHO0)/CV
      dmmhtfnc=pzero
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMAMA(BM,TWOG,TWOG1,EDEVMAGSQ)
C     Purpose:
#include "precision.Blk"
      PARAMETER (PONE=1.D0,PTWO=2.D0,PTHREE=3.D0,PFIVE=5.D0,PSIX=6.D0)
      PARAMETER (PNINE=9.D0)
      PARAMETER (PI=3.1415926535897932384626433832795028841971693993D0)
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      XISOMAG=SQRT(PNINE*BM**2 + PFIVE*TWOG**2)
      XMAG=SQRT(PNINE*BM**2+PFIVE*TWOG**2+PSIX*TWOG1**2*EDEVMAGSQ)
C     Scalar measure of anisotropy
      DMMAMA= PTWO/PI*ACOS(XISOMAG/XMAG)
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMHRD(DUM1,DUM2,DUM3,DUM4)
      DMMHRD=0.D0*(DUM1+DUM2+DUM3+DUM4)
      RETURN
      END

#ifdef KERLEY_EOS_RTNS
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE EOSUI(UIOLD,UI)
C***********************************************************************
C     PURPOSE: Create new UI array with properties ordered as required
C     by the Kerley EOS
C
C input
C -----
C    UIOLD: Original property array
C
C output
C -----
C    UI: Newly re-ordered property array
C
C  MODIFICATION HISTORY
C  yymmdd|usrname|what was done
C  ------ --- -------------
C  100422|tjfulle|created routine
C
#ifdef TJF_IMPLNONE
      IMPLICIT NONE
#else
#include "precision.Blk"
#endif
C............................................................parameters
#include "elastic_plastic_pnt.Blk"
C.................................................................common
C.................................................................passed
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION UIOLD,UI
#endif
      DIMENSION UIOLD(*),UI(*)
C..................................................................local
CccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccEOSUI
      UI(1)= UIOLD(IPRHO0)   ! - initial density (required)
      UI(2)= UIOLD(IPT0)     ! - initial temperature
      UI(3)= UIOLD(IPC0)     ! - sound speed (required)
      UI(4)= UIOLD(IPS1)     ! - coefficient of linear term US-UP fit
      UI(5)= UIOLD(IPGP)     ! - Gruneisen parameter
      UI(6)= UIOLD(IPCV)     ! - heat capacity (required)
      UI(7)= UI(6)*UI(2)     ! - shift in energy zero (see below)
      UI(8)= 0.D0            ! - initial porous density
      UI(9)= 0.D0            ! - crushup pressure
      UI(10)= 0.D0           ! - pressure at elastic limit
      UI(11)= 3670.D0        ! - sound speed in foam
      UI(12)= 1.D0           ! - number of subcycles in time step
      UI(13)= 0.D0           ! - coefficient of quad. term in US-UP fit
      UI(14)= 1.D0           ! - used for 2-state and MP models
      UI(15)= UIOLD(IPRHO0)  ! - alias for R0
      UI(16)= UIOLD(IPT0)    ! - alias for T0
      UI(17)= UIOLD(IPS1)    ! - alias for S1
      UI(18)= UIOLD(IPGP)    ! - alias for G0
      UI(19)= 0.D0           ! - coefficient of low-pressure term
      UI(20)= 1.D0           ! - constant in low-pressure term
      UI(21)= 1.D0           ! - power of compression in low-pressure term
      UI(22)= 2.D0           ! - power for alpha integration
      RETURN
      END
C
C
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION EXPS(ARG)
C***********************************************************************
C     PURPOSE: Specialized version of exponential to prevent underflow
C              replacement of EXP(ARG) by zero when ARG is an
C              extraordinarily large negative number.  For all practical
C              purposes, EXPS may be regarded as equivalent to EXP.
C
C input
C -----
C    ARG: Argument for exponential
C
C output
C -----
C    EXPS: Exponential of ARG
C
C
      IMPLICIT NONE
C.............................................................parameters
      DOUBLE PRECISION EUNDERFLOW,EOVERFLOW,PONE
      PARAMETER (PONE=1.D0)
      PARAMETER (EUNDERFLOW=-34.53877639491*PONE)
      PARAMETER (EOVERFLOW=92.1034037*PONE)
C.................................................................common
C.................................................................passed
      DOUBLE PRECISION ARG,EXPS
C
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
C     EXPS(ARG)       = EXP(ARG)
      EXPS= EXP(MIN(MAX(ARG,EUNDERFLOW),EOVERFLOW))
      RETURN
      END ! FUNCTION EXPS
#endif
