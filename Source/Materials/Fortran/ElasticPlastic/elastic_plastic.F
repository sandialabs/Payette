      SUBROUTINE DIAMM_CALC(NBLK,NINSV,DT,UI,SIG,D,SV,USM)
C***********************************************************************
C
C     Description:
C           Drucker-Prager plasticity model with elastic strain induced
C           anisotropy.
C
C***********************************************************************
C
C     input arguments
C     ===============
C      NBLK       int                   Number of blocks to be processed
C      NINSV      int                   Number of internal state vars
C      DT         dp                    Current time increment
C      UI       dp,ar(nprop)            User inputs
C      D          dp,ar(6)              Strain increment
C
C     input output arguments
C     ======================
C      STRESS   dp,ar(6)                stress
C      SV       dp,ar(ninsv)            state variables
C
C     output arguments
C     ================
C      USM      dp                      uniaxial strain modulus
C
C***********************************************************************
C
C      stresss and strains, plastic strain tensors
C          11, 22, 33, 12, 23, 13
C
C***********************************************************************
      IMPLICIT NONE
C
C.............................................................parameters
#include "elastic_plastic_pnt.Blk"
      DOUBLE PRECISION PTHIRD,PHALF,P3HALF,P2THIRD
      DOUBLE PRECISION PZERO,PONE,PTWO,PTHREE,ROOT2
      DOUBLE PRECISION TOL1M10,YLDTOL
      PARAMETER (PTHIRD=0.333333333333333333333333333333333333333333D0)
      PARAMETER (P2THIRD=0.66666666666666666666666666666666666666667D0)
      PARAMETER (  ROOT2=0.141421356237309504880168872420969807856967D1)
      PARAMETER (PHALF=0.5D0,P3HALF=1.5D0)
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTWO=2.D0,PTHREE=3.D0)
      PARAMETER (TOL1M10= 1.0D-10,YLDTOL=1.0D-3)
C.................................................................common
#include "elastic_plastic_prop.h"
C.................................................................passed
      INTEGER NBLK,NINSV
      DOUBLE PRECISION SV,SIG,USM,UI,D,DT
      DIMENSION SV(NINSV),SIG(6),USM(NBLK)
      DIMENSION UI(*),D(6)
C..................................................................local
      INTEGER IJ,I
      DOUBLE PRECISION DUMA
      DOUBLE PRECISION DUM,RJN
      DOUBLE PRECISION BM,SM,THREEK,TWOG,YF
      DOUBLE PRECISION T,TMLT,U,RHO,DTE,DTP
      DOUBLE PRECISION DELTA,FAC
      DOUBLE PRECISION TRDTAU,SINVDTAU
      DOUBLE PRECISION S,TAUN,QSTAUN,QSTAUT,QSTAUP,TAUP
      DOUBLE PRECISION QSTH
      DOUBLE PRECISION DTAUT,DTAU,DTAUISO,DTAUDEV,APART
      DOUBLE PRECISION DE,DEISO,DEDEV,DEP,DEE
      DOUBLE PRECISION EN,EISO,EDEV,DEPDEV,DEPISO,TRDEP,SINVDEP
      DOUBLE PRECISION TRDE,SINVDE,BETA1,BETA2,DEQPS
      DOUBLE PRECISION RNN,RMM,RMAGM,RMAGN,P,Z,TRM,A
      DOUBLE PRECISION RMMI,RMMD,DCSP
      DOUBLE PRECISION DFDI1,DFDJ2,DGDI1,DGDJ2
      DOUBLE PRECISION ALPHA1,ALPHA2,DDD,BETA,Y1,Y2,ZETA
      DOUBLE PRECISION H
C     symmetric stress tensors
      DIMENSION S(6),TAUN(6),TAUP(6)
      DIMENSION DTAUT(6),DTAU(6),DTAUISO(6),DTAUDEV(6)
      DIMENSION QSTAUN(6),QSTAUT(6),QSTAUP(6)
      DIMENSION QSTH(6)
C     symmetric strain tensors
      DIMENSION DE(6),DEDEV(6),DEISO(6),DEE(6)
      DIMENSION DEP(6),DEPDEV(6),DEPISO(6)
C     misc symmetrc tensors
      DIMENSION DUMA(6)
      DIMENSION DELTA(6),RMMI(6),RMMD(6),APART(6)
      DIMENSION RNN(6),RMM(6),P(6),Z(6),A(6)
      SAVE DELTA
      DATA DELTA/PONE,PONE,PONE,PZERO,PZERO,PZERO/
C     Needed for induced anisotropy
      DIMENSION EN(6),EISO(6),EDEV(6)
C...................................................................data
      LOGICAL INELASTIC
C...............................................................external
      DOUBLE PRECISION DMMTR,DMMDBD,DMMEN,DMMYLD
      DOUBLE PRECISION DMMRJ2,DMMTS,DMMTMG
      DOUBLE PRECISION DMMAMA,DMMHRD
C....................................................statement functions
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc DIAMM_CALC
C
      IF(NINSV.NE.NDMMISV)CALL BOMBED('bad ninsv sent to Diamm')
      INELASTIC=.FALSE.

C     Copy passed field arguments that are subject to
C     being updated into local arrays and initialize other variables

C--------------------------------------------------- start initial setup
C     Put user inputs in common variables
      CALL DMMPAR(UI)

C     Quasistatic and dynamic Kirchhoff stress, strain increment,
C     Elastic strain
      RJN= SV(KRJ)
      DO IJ=1,6
         TAUN(IJ)= SIG(IJ)*SV(KRJ)
         QSTAUN(IJ)= SV(KQSSIG+IJ)*SV(KRJ)
         DE(IJ)= D(IJ)*DT
         EN(IJ)= SV(KE+IJ)
      ENDDO

      CALL DMMTD(DE,DEISO,DEDEV,TRDE, SINVDE)
      CALL DMMTD(EN, EISO, EDEV, BETA1,BETA2)

C     Initialize variables that change with deformation
      T= SV(KT)
      U= SV(KEU)
      TMLT= DMMTS(T) ! J-C homologous temperature
      RHO= SV(KRHO)
C----------------------------------------------------- end initial setup

C--------------------------------------------------- begin stress update
C
C     Calculate the stress-invariant dependent elastic props
C
      CALL DMMMOD(UI,U,RHO,T,BETA1,QSTAUN,BM,SM)
      TWOG= PTWO*SM
      THREEK= PTHREE*BM
      USM(NBLK)= PTHIRD*(PTWO*TWOG+THREEK)
C
C     Trial stress
C
      DO IJ=1,6
         APART(IJ)= TWOG1*(TRDE*EDEV(IJ) + DMMDBD(DEDEV,EDEV)*DELTA(IJ))
         DTAUT(IJ)= THREEK*DEISO(IJ) + TWOG*DEDEV(IJ) + APART(IJ)
         DUMA(IJ)=DTAUT(IJ)-APART(IJ)
         QSTAUT(IJ)= QSTAUN(IJ) + DTAUT(IJ)
         QSTH(IJ)= QSTAUT(IJ)
      ENDDO
C
C
C     Check trial stress
C
C
      YF= DMMYLD(QSTAUT,SV(KEQPS),T)
      IF(YF.LT.YLDTOL)THEN
C
C
C     ELASTIC
C
C
         DO IJ=1,6
            DEP(IJ)= PZERO
            DEE(IJ)= DE(IJ)
            QSTAUP(IJ)= QSTAUT(IJ)
            SV(KE+IJ)= SV(KE+IJ) + DEE(IJ) !total elastic strain
            DTAU(IJ)= DTAUT(IJ)
         ENDDO
         DEQPS= PZERO

      ELSE
         INELASTIC=.TRUE.
C
C
C     PLASTIC
C
C
         IF(WANTSELASTIC)CALL BOMBED('Plastic loading has occurred')
         IF(PRESDEPYLD.AND.DMMTR(QSTAUT).GT.PZERO)THEN
C     Tensile
C     We are at a place where the stress might be returned to a vertex.
C     Right now I am really only concerned about compressive behavior, so
C     the following just sets the tensile side of the yield surface
C     to behave like Von Mises.  Is it wrong and not as elegant as true
C     vertex theory?  YES!  Does it work? As advertised!
            A2=PZERO
            A3=PZERO
            A4=PZERO
            A4G=PZERO
         ENDIF


C     Apply oblique return to put stress on yield surface

C
C     Newton iterations to find magnitude of projection from the trial
C     stress state to the yield surface.
C
         DO I=1,25
C
C     Yield normal and flow direction
C
            DFDI1= (A2*A3*EXP(A2*DMMTR(QSTAUT))+A4)*TMLT
            DFDJ2= PHALF/DMMRJ2(QSTAUT)
            RMAGN= SQRT(PHALF + PTHREE*DFDI1**2)

            DGDI1= (A2*A3*EXP(A2*DMMTR(QSTAUT))+A4G)*TMLT
            DGDJ2= PHALF/DMMRJ2(QSTAUT)
            RMAGM= SQRT(PHALF + PTHREE*DGDI1**2)

            CALL DMMDEV(QSTAUT,S)
            DO IJ=1,6
               RNN(IJ)= (DFDI1*DELTA(IJ) + DFDJ2*S(IJ))/RMAGN
               RMM(IJ)= (DGDI1*DELTA(IJ) + DGDJ2*S(IJ))/RMAGM
            ENDDO
C
C     Z,A, and P tensors
C
            CALL DMMTD(RMM,RMMI,RMMD,TRM,DUM)
            DO IJ=1,6
               Z(IJ)= GP*DMMDBD(QSTAUN,RMM)*DELTA(IJ)*pzero
               APART(IJ)=TWOG1*(DMMDBD(EDEV,RMM)*DELTA(IJ)+TRM*EDEV(IJ))
               A(IJ)= THREEK*RMMI(IJ) + TWOG*RMMD(IJ) + APART(IJ)
               P(IJ)= A(IJ) + Z(IJ)
            ENDDO
            FAC= SQRT(DMMTMG(QSTAUT)/DMMTMG(P))
            DO IJ=1,6
               P(IJ)= FAC*P(IJ)
            ENDDO
C
C     Hardening modulus -- returns zero for now
C
            H= DMMHRD(TAUN,RMM,SV(KEQPS),T)
C
C     Apply the Newton-Raphson step
C
            BETA = -YF/DMMDBD(RNN,P)
C
C     Improved estimates for the test stress
C
            DO IJ=1,6
               QSTAUP(IJ)= QSTAUT(IJ) + BETA*P(IJ)
            ENDDO
C
C     Check for convergence
C
            IF(ABS(BETA).LT.TOL1M10.AND.YF.LT.YLDTOL)GO TO 20
C
C     If not converged, update the trial stress and yield function
C
            DO IJ=1,6
               QSTAUT(IJ)= QSTAUP(IJ)
            ENDDO
            YF= DMMYLD(QSTAUT,SV(KEQPS),T)

         ENDDO
C
C
      CALL BOMBED('Newton iterations failed')

 20      CONTINUE
C
C
C     Quasi-static stress now in QSTAUP
C
C

C
C     Updated quasistatic stress increment
C
         DO IJ=1,6
            DTAU(IJ)= QSTAUP(IJ) - QSTAUN(IJ)
         ENDDO
         CALL DMMTD(DTAU,DTAUISO,DTAUDEV,TRDTAU,SINVDTAU)

C
C     Elastic strain increment and updated elastic strain
C
         IF(ANISO)THEN
            ZETA= (BETA2*TWOG1**2)/SM/BM
            DDD= THREEK*TWOG*(PONE-ZETA)/(TWOG1**2)
            Y1= DMMTR(DTAU)
            Y2= DMMDBD(EDEV,DTAU)
            ALPHA1= PONE/DDD*(P2THIRD*BETA2*Y1/BM - Y2/TWOG1)
            ALPHA2= PONE/DDD*(P3HALF*Y2/SM - Y1/TWOG1)
         ELSE
            ALPHA1= PZERO
            ALPHA2= PZERO
         ENDIF
         DO IJ=1,6
            DEE(IJ)= PONE/THREEK*DTAUISO(IJ) + PONE/TWOG*DTAUDEV(IJ)
     &           + ALPHA1*DELTA(IJ) + ALPHA2*EDEV(IJ)
            SV(KE+IJ)= SV(KE+IJ) + DEE(IJ) !total elastic strain
         ENDDO
C
C     Plastic strain increment
C
         DO IJ=1,6
            DEP(IJ)= DE(IJ)-DEE(IJ)
         ENDDO
C
C
      ENDIF   ! End plastic return
C
C
C
C
C
C
C
C
C

C
C     Update equivalent plastic strain
C
      CALL DMMTD(DEP,DEPISO,DEPDEV,TRDEP,SINVDEP)
      DEQPS = SQRT(DMMDBD(DEPDEV,DEPDEV)+PTHIRD*TRDEP**2)
      IF(DEQPS.LT.1.D-16)THEN
         DO IJ=1,6
            DEP(IJ) = PZERO
         ENDDO
         DEQPS = PZERO
      ENDIF
      SV(KEQPS)= SV(KEQPS) + DEQPS
C
C     Update temperature
C
      SV(KPW)= DMMEN(QSTAUP,DEP,RHO0) ! plastic work
      DTE= -SV(KT)*GP*TRDE      ! elastic tmpr increment
      DTP= UI(IPTQC)*SV(KPW)/CV ! plastic tmpr increment
      SV(KT)=MIN(TM,SV(KT)+DTE+DTP)
C
C
C
C
C
C
C     Update quasistatic ISVs
C
      SV(KRHO)= SV(KRHO)*EXP(-TRDE)
      SV(KRJ)= RHO0/SV(KRHO)
      SV(KEU)= SV(KEU) + DMMEN(QSTAUP,DE,RHO0)
      SV(KCS)= SQRT(BM/RHO0)
      SV(KEQPV)= SV(KEQPV)+DMMTR(DEP)
C
C
C
C
C
C
C

C
C
C     Calculate the dynamic overstress TAUP
C
C
      CALL OVERSTRESS(DT,SV,DMMDBD(D,D),TAUN,DTAUT,QSTAUN,QSTAUP,TAUP)
C
C
C     Dynamic overstress now in TAUP
C
C

C
C     State variables at end of step
C
      SV(KEQDOT)= DMMDBD(D,D)
      CALL DMMDEV(SV(KE+1),EDEV)
      SV(KAM)= DMMAMA(BM,TWOG,TWOG1,DMMDBD(EDEV,EDEV))
C
C     Update energy to include work increment from dynamic stress
C
      SV(KEU)= SV(KEU)
     &     +(DMMEN(TAUP,D,RHO0)-DMMEN(QSTAUP,D,RHO0))*DT
C
C     Update passed arguments
C
C     Dynamic and quasistatic stress arrays
C
      DO IJ=1,6
         SIG(IJ)= TAUP(IJ)/RJN
         SV(KQSSIG+IJ)= QSTAUP(IJ)/RJN
      END DO
      SV(KROOTJ2)= DMMRJ2(SV(KQSSIG+1))
      SV(KI1)= DMMTR(SV(KQSSIG+1))

      IF(INELASTIC)THEN
C     Get final return direction
         DCSP= DMMTMG(DEP)
         DO IJ=1,6
            P(IJ)= (QSTH(IJ) - QSTAUP(IJ))/DCSP
         ENDDO
      ELSE
         DCSP= PZERO
         DO IJ=1,6
            P(IJ)= PZERO
         ENDDO
      ENDIF
      SV(KF1)= DMMTR(QSTAUN)/RJN !3
      SV(KF2)= DMMRJ2(QSTAUN)/RJN !4
      SV(KF3)= DMMTR(DTAUT)/RJN !5
      SV(KF4)= DMMRJ2(DTAUT)/RJN !6
      SV(KF5)= DMMTR(DTAU)/RJN  !7
      SV(KF6)= DMMRJ2(DTAU)/RJN !8
      SV(KF7)= DMMTR(QSTH)/RJN  !9
      SV(KF8)= DMMRJ2(QSTH)/RJN !10
      SV(KF9)= -DCSP*DMMTR(P)/RJN !11
      SV(KF10)=-DCSP*DMMRJ2(P)/RJN !12

      RETURN
      END
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
      IMPLICIT NONE
C............................................................parameters
      DOUBLE PRECISION PZERO,PONE,PTHREE,PTHIRD,PHALF
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTHREE=3.D0)
      PARAMETER (PTHIRD=0.333333333333333333333333333333333333333333D0)
      PARAMETER (PHALF=0.5D0)
C.................................................................passed
      DOUBLE PRECISION A,FINVA,SINVA,AISO,ADEV
      DIMENSION A(6),AISO(6),ADEV(6)
C..................................................................local
      INTEGER IJ
      DOUBLE PRECISION TMP,DELTA
      DIMENSION DELTA(6)
      SAVE DELTA
      DATA DELTA/PONE,PONE,PONE,PZERO,PZERO,PZERO/
C...............................................................external
      DOUBLE PRECISION DMMDBD
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
      IMPLICIT NONE
C............................................................parameters
      DOUBLE PRECISION PZERO,PONE,PTHREE,PTHIRD
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTHREE=3.D0,PTHIRD=PONE/PTHREE)
C.................................................................passed
      DOUBLE PRECISION A,ADEV
      DIMENSION A(6),ADEV(6)
C..................................................................local
      INTEGER IJ
      DOUBLE PRECISION TMP,DELTA
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
      IMPLICIT NONE
C.............................................................parameters
      DOUBLE PRECISION PZERO,PTWO,PTHREE,PSIX,ROOT3,TOOR3,ROOT23,PTHIRD
      PARAMETER (PZERO=0.0D0,PTWO=0.2D1,PTHREE=0.3D1,PSIX=6.0D0)
      PARAMETER (PTHIRD=0.3333333333333333333333333333333333333333333D0)
      PARAMETER (  ROOT3=0.173205080756887729352744634150587236694281D1)
      PARAMETER (  TOOR3=0.577350269189625764509148780501957455647602D0)
      PARAMETER ( ROOT23=0.816496580927726032732428024901963797321982D0)
#include "elastic_plastic_pnt.Blk"
C.................................................................common
#include "elastic_plastic_prop.h"
C.................................................................passed
      DOUBLE PRECISION UI
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
      IMPLICIT NONE
C.............................................................parameters
      DOUBLE PRECISION TOL1M10,TOOR3,HUGE
      PARAMETER (TOL1M10=1.D-10,HUGE=1.D90)
      PARAMETER (  TOOR3=0.577350269189625764509148780501957455647602D0)
C.................................................................common
C.................................................................passed
      DOUBLE PRECISION A,AZ,AR,ER
      DIMENSION A(6),ER(6)
C...............................................................external
      DOUBLE PRECISION DMMTR,DMMTMG
C..........................................................local scalars
      INTEGER IJ
      DOUBLE PRECISION DUM
C...........................................................local arrays
      DOUBLE PRECISION S
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
      IMPLICIT NONE
C............................................................parameters
#include "elastic_plastic_pnt.Blk"
      DOUBLE PRECISION PZERO,PHALF,PONE,PTWO,PTHREE,PTEN,PTHIRD
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTWO=2.D0,PTHREE=3.D0,PTEN=10.D0)
      PARAMETER (PHALF=0.5D0)
      PARAMETER (PTHIRD=0.333333333333333333333333333333333333333333D0)
C.................................................................common
#include "elastic_plastic_prop.h"
C.................................................................passed
      DOUBLE PRECISION UI,U,RHO,T,EV,BM,SM,TAU
      DIMENSION UI(*),TAU(6)
C..................................................................local
      INTEGER IDK,IDG
      DOUBLE PRECISION P
      DOUBLE PRECISION CS,RJFAC
#ifdef KERLEY_EOS_RTNS
      DOUBLE PRECISION S,UIMG
      DIMENSION S(9),UIMG(22)
#endif
C...................................................................data
C...............................................................external
      DOUBLE PRECISION DMMTR
C....................................................statement functions
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc DMMMOD
      IDK=NINT(UI(IPIDK))
      IDG=NINT(UI(IPIDG))
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
      IMPLICIT NONE
C
C.............................................................parameters
#include "elastic_plastic_pnt.Blk"
      DOUBLE PRECISION PTHIRD,PHALF,PZERO,PONE,PSIX,TOL1M20,EUNDERFLOW
      PARAMETER (PTHIRD=0.3333333333333333333333333333333333333333333D0)
      PARAMETER (PHALF=0.5D0)
      PARAMETER (PZERO=0.D0,PONE=1.D0,PSIX=6.D0)
      PARAMETER (TOL1M20=1.D-20)
      PARAMETER (EUNDERFLOW=-34.53877639491D0*PONE)
C.................................................................common
#include "elastic_plastic_prop.h"
C.................................................................passed
      DOUBLE PRECISION DT,EQDOT,SIGOLD,QSSIGOLD,QSSIG,DSIGT,SIG,SV
      DIMENSION SIGOLD(6),QSSIGOLD(6),QSSIG(6),DSIGT(6),SIG(6),SV(*)
C..................................................................local
      INTEGER IJ,I
      DOUBLE PRECISION SIGV,DUM,TCHAR,RHI,RLO,RMID,RAT
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
      IMPLICIT NONE
      DOUBLE PRECISION A,DMMTMG,DMMDBD
      DIMENSION A(6)
      DMMTMG=SQRT(DMMDBD(A,A))
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMDBD(A,B)
C     Purpose: Double dot product of second order tensors A and B
      IMPLICIT NONE
      DOUBLE PRECISION A,B,PTWO,DMMDBD
      PARAMETER (PTWO=2.D0)
      DIMENSION A(6),B(6)
      DMMDBD=     A(1)*B(1)+A(2)*B(2)+A(3)*B(3)
     $    + PTWO*(A(4)*B(4)+A(5)*B(5)+A(6)*B(6))
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMTR(A)
C     Purpose: Trace of second order tensor A
      IMPLICIT NONE
      DOUBLE PRECISION A,DMMTR,TOL,PZERO
      DIMENSION A(6)
      PARAMETER (TOL=1.D-15,PZERO=0.D0)
      DMMTR=A(1)+A(2)+A(3)
      IF(ABS(DMMTR).LT.TOL)DMMTR=PZERO
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMRJ2(A)
C     Purpose: Square root of second invariant of second order tensor A
      IMPLICIT NONE
      DOUBLE PRECISION A,AD,PHALF,DMMDBD,DMMRJ2
      DIMENSION A(6),AD(6)
      PARAMETER (PHALF=0.5D0)
      CALL DMMDEV(A,AD)
      DMMRJ2=SQRT(PHALF*DMMDBD(AD,AD))
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMEN(A,B,RHO)
C     Purpose: Energy inc. for stress A, strain inc. B and density RHO
      IMPLICIT NONE
      DOUBLE PRECISION A,RHO,B,DMMDBD,DMMEN
      DIMENSION A(6),B(6)
      DMMEN= DMMDBD(A,B)/RHO
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMTS(T)
C     Purpose: Johnson-Cook homologous temperature
      IMPLICIT NONE
#include "elastic_plastic_prop.h"
      DOUBLE PRECISION PZERO, PONE, TARG, DMMTS, T, DMMTMULT
      PARAMETER(PZERO=0.D0,PONE=1.D0)
      TARG= MAX(T-T0,PZERO)
      DMMTS= PONE-(TARG/(TM-T0))**XP
      dmmtmult= pone
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMYLD(TAU,EQPS,T)
C     Purpose: Value of yield function
      IMPLICIT NONE
#include "elastic_plastic_prop.h"
      DOUBLE PRECISION EQPS,T, DMMTR, DMMTS
      DOUBLE PRECISION ROOT23,TAU,RI1,FF,DMMYLD,DMMRJ2
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
      IMPLICIT NONE
#include "elastic_plastic_prop.h"
      DOUBLE PRECISION PZERO, D1, D2, D3, TARG, DMMDYLDFNCDT
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
      IMPLICIT NONE
      DOUBLE PRECISION PZERO, A3, A4, EQPS, RMS, HY, DMMHYFNC
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
      IMPLICIT NONE
      DOUBLE PRECISION PZERO,TAUN,RM,RHO0,CV,DMMHTFNC,DMMEN
      PARAMETER (PZERO=0.D0)
      DIMENSION TAUN(6),RM(6)
      DMMHTFNC= DMMEN(TAUN,RM,RHO0)/CV
      dmmhtfnc=pzero
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      FUNCTION DMMAMA(BM,TWOG,TWOG1,EDEVMAGSQ)
C     Purpose:
      IMPLICIT NONE
      DOUBLE PRECISION PONE,PTWO,PTHREE,PFIVE,PSIX,PNINE,PI
      DOUBLE PRECISION XISOMAG,DMMAMA,BM,TWOG,TWOG1,EDEVMAGSQ,XMAG
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
      DOUBLE PRECISION DMMHRD,DUM1,DUM2,DUM3,DUM4
      DMMHRD=0.D0*(DUM1+DUM2+DUM3+DUM4)
      RETURN
      END

C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMCHK(UI)
C
C***********************************************************************
C     REQUIRED MIG DATA CHECK ROUTINE
C     Checks validity of user inputs for DMM model.
C     Sets defaults for unspecified user input.
C     Adjusts user input to be self-consistent.
C
C     input
C     -----
C       UI: user input as read and stored by host code.
C
C       Upon entry, the UI array contains the user inputs EXACTLY
C       as read from the user.  These inputs must be ordered in the
C       UI array as indicated in the file kmmpnt.Blk.
C       See kmmpnt.Blk for parameter definitions and keywords.
C
C
C    Other output
C    ------------
C
C***********************************************************************
C  author:  Rebecca Brannon
C
C  yymmdd:usernam:   m o d i f i c a t i o n
C  ---------------------------------------------------------------------
C  090713:tim fuller:Created original data check
C
      IMPLICIT NONE
#include "elastic_plastic_pnt.Blk"
C
      DOUBLE PRECISION UI,PZERO,PONE,PTWO,PTHREE,HUGE,ROOT3,TOOR3
      DIMENSION UI(*)
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTWO=2.D0,PTHREE=3.D0)
      PARAMETER (HUGE=1.D80)
      PARAMETER (  ROOT3=0.173205080756887729352744634150587236694281D1)
      PARAMETER (  TOOR3=PONE/ROOT3)
C
C  ...local
      CHARACTER*6 IAM
      PARAMETER( IAM = 'DMMCHK' )
#ifdef KERLEY_EOS_RTNS
      DOUBLE PRECISION A,CS,GP,S1
      DIMENSION A(8)
#endif
      LOGICAL DEJAVU
      DATA DEJAVU/.FALSE./

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c     avoid compiler warning of unused dummy arguments
c
      DEJAVU=(NINT(UI(IPDEJAVU)).NE.0)
C
      IF(.NOT.DEJAVU)THEN
        CALL LOGMES('############# DMM build 100222')
      ENDIF
C
C     Check validity of user-supplied model parameters
C
      IF(UI(IPRHO0).LT.PZERO)CALL FATERR(IAM,'R0 must be positive')
      IF(UI(IPB0).LE.PZERO)THEN
         IF(UI(IPC0).LE.PZERO)THEN
            CALL FATERR(IAM,'B0 or C0 must be positive')
         ELSE
            UI(IPB0)=UI(IPRHO0)*UI(IPC0)*UI(IPC0)
         ENDIF
      ELSE
         IF(UI(IPC0).LE.PZERO)THEN
            UI(IPC0)=SQRT(UI(IPB0)/UI(IPRHO0))
         ENDIF
      ENDIF
      IF(UI(IPG0).LE.PZERO)CALL FATERR(IAM,'G0 must be positive')
      IF(UI(IPA1).LT.PZERO)CALL FATERR(IAM,'A1 must be nonnegative')
      IF(UI(IPA2).LT.PZERO)CALL FATERR(IAM,'A2 must be nonnegative')
      IF(UI(IPA3).LT.PZERO)CALL FATERR(IAM,'A3 must be nonnegative')
      IF(UI(IPA4).LT.PZERO)CALL FATERR(IAM,'A4 must be nonnegative')
      IF(UI(IPA4PF).LT.PZERO)CALL FATERR(IAM,'A4PF must be nonnegative')

c     Set defaults
      IF(UI(IPA4PF).EQ.PZERO)UI(IPA4PF)=UI(IPA4)  !default A4PF = A4

c     Convert derivative of G w.r.t. pressure to derivative w.r.t.
c     elastic volume change
      IF(.NOT.DEJAVU)THEN
         IF(UI(IPG1).LE.PZERO)THEN
            UI(IPG1)=PZERO
            UI(IPAN)=PZERO
         ENDIF
         IF(UI(IPB1).LE.PZERO)UI(IPB1)=PZERO
         UI(IPB1)= -UI(IPB1)*UI(IPB0)
         UI(IPG1)= -UI(IPG1)*UI(IPB0)
      ENDIF

C     Get Drucker Prager coefficients if compressive strength is given
      IF(UI(IPA1).LT.HUGE.AND.UI(IPSC).GT.PZERO)THEN
         IF(UI(IPA4).LE.PZERO)THEN
            UI(IPA4)=TOOR3*(UI(IPSC)-UI(IPA1))/(UI(IPSC)+UI(IPA1))
c     UI(IPA1)=PTWO/ROOT3*UI(IPSC)*UI(IPA1)/(UI(IPSC)+UI(IPA1))
         ENDIF
      ENDIF
      IF(UI(IPT0).LE.PZERO)UI(IPT0)=298.D0
      IF(UI(IPRHO0).LE.PZERO)UI(IPRHO0)=PONE
      IF(UI(IPTM).LE.PZERO)UI(IPTM)=1.D99
      IF(UI(IPCV).LE.PZERO)UI(IPCV)=PONE
      IF(UI(IPGP).LE.PZERO)UI(IPGP)=PZERO
      IF(UI(IPXP).LE.PZERO)UI(IPXP)=PONE
      IF(UI(IPTQC).LE.PZERO)UI(IPTQC)=PONE

      UI(IPIDK) = PZERO
      UI(IPIDG) = PZERO

      IF(.NOT.DEJAVU)THEN
         IF(UI(IPA4PF).EQ.UI(IPA4))THEN
            CALL LOGMES(IAM//': FYI, This material is associative')
         ELSE
            CALL LOGMES(IAM//': FYI, This material is nonassociative')
         ENDIF
        CALL LOGMES('############# DMM data check complete')
        UI(IPDEJAVU)=PONE
        IF(PTHREE*UI(IPB0).LT.PTWO*UI(IPG0))THEN
           CALL LOGMES
     & (IAM//' Warning: neg Poisson (to avoid warning, set 3*B0>2*G0)')
        ENDIF
      ENDIF
      RETURN
      END !SUBROUTINE DMMCHK
C
C
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE DMMRXV(UI, NX, NAMEA, KEYA, RINIT, RDIM, IADVCT, ITYPE)
C**********************************************************************
C     REQUESTED EXTRA VARIABLES FOR KAYENTA
C
C     This subroutine creates lists of the internal state variables
C     needed for DMM. This routine merely sends a
C     LIST of internal state variable requirements back to the host
C     code.   IT IS THE RESPONSIBILITY OF THE HOST CODE to loop over
C     the items in each list to actually establish necessary storage
C     and (if desired) set up plotting, restart, and advection
C     control for each internal state variable.
C
C     called by: host code after all input data have been checked
C
C     input
C     -----
C          UI = user input array
C
C     output
C     ------
C          NX = number of extra variables                    [DEFAULT=0]
C       NAMEA = single character array created from a string array
C               (called NAME) used locally in this routine to register
C               a descriptive name for each internal state variable.
C        KEYA = single character array created from a string array
C               (called KEY) used locally in this routine to register
C               a plot keyword for each internal state variable.
C          | Note: NAMEA and KEYA are created from the local variables |
C          | NAME and KEY by calls to the subroutine TOKENS, which     |
C          | is a SERVICE routine presumed to ALREADY exist within the |
C          | host code (we can provide this routine upon request).     |
C          | "NAME" is a fortran array of strings. "NAMEA" is a one    |
C          | dimensional array of single characters. For readability,  |
C          | most of this subroutine writes to the NAME array. Only at |
C          | the very end is NAME converted to NAMEA by calling a      |
C          | the utility routine TOKENS. The KEY array is similarly    |
C          | converted to KEYA.  These conversions are performed       |
C          | because host codes written in C or C++ are unable to      |
C          | process FORTRAN string arrays. Upon request, we can       |
C          | provide a utility routine that will convert BACK to       |
C          | FORTRAN string arrays if your host code is FORTRAN.       |
C          | Likewise, we can provide a C++ routine that will allow    |
C          | parsing the single-character arrays to convert them back  |
C          | to strings if your code is C++. Alternatively, you can    |
C          | simply ignore the NAMEA and KEYA outputs of this routine  |
C          | if your host code does not wish to establish plotting     |
C          | information.                                              |
C
C       RINIT = initial value for each ISV               [DEFAULT = 0.0]
C        RDIM = physical dimension exponents             [DEFAULT = 0.0]
C               This variable is dimensioned RDIM(7,*) for the 7 base
C               dimensions (and * for the number of extra variables):
C
C                      1 --- length
C                      2 --- mass
C                      3 --- time
C                      4 --- temperature
C                      5 --- discrete count
C                      6 --- electric current
C                      7 --- luminous intensity
C
C                Suppose, for example, that an ISV has units of stress.
C                Dimensionally, stress is length^(1) times mass^(-1)
C                times time^(-2). Therefore, this routine would return
C                1.0, -1.0, and -2.0 as the first three values of the
C                RDIM array. Host codes that work only in one unit
C                set (e.g., SI) typically ignore the RDIM output.
C
C      IADVCT = advection option                           [DEFAULT = 0]
C                    = 0 advect by mass-weighted average
C                    = 1 advect by volume-weighted average
C                    = 2 don't advect
C            The advection method will often be ignored by host codes.
C            It is used for Eulerian codes and for Lagrangian codes that
C            re-mesh (and therefore need guidance about how to "mix"
C            internal state variables). Note: a value of 2 implies that
C            the ISV is output only.
C
C        ITYPE = variable type (see migtionary preface)    [DEFAULT = 1]
C                  1=scalar
C                  6=2nd-order symmetric tensor
C        The component ordering for ITYPE=6 is 11, 22, 33, 12, 23, 31.
C        Consequently, the 11 component is the first one to be requested
C        in tensor lists, and its IFLAG is set to 6. To indicate that
C        subsequent ISVs are the remaining components of the same tensor,
C        the next five ISVs are given an IFLAG value of -6.
C        Host codes that don't change basis can ignore ITYPE.
C
C***********************************************************************
C
C  author:  Rebecca Brannon
C
C    who    yymmdd  M O D I F I C A T I O N
C  -------  ------  ----------------------------------------------------
C  rmbrann  030809  Created original extra variable routine
C

      IMPLICIT NONE
#include "elastic_plastic_pnt.Blk"
C
      DOUBLE PRECISION PZERO,PONE,PTWO,PTHREE
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTWO=2.D0,PTHREE=3.D0)
      INTEGER MMCN,MMCK,MNUNIT,MMCNA,MMCKA
      INTEGER NX, IADVCT, ITYPE
      DOUBLE PRECISION UI,RINIT,RDIM
      PARAMETER (MMCN=50,MMCK=10,MNUNIT=7)
      PARAMETER (MMCNA=NDMMISV*MMCN,MMCKA=NDMMISV*MMCK)
C
      CHARACTER*(MMCN) NAME(NDMMISV)
      CHARACTER*(MMCK) KEY(NDMMISV)
      CHARACTER*1 NAMEA(*), KEYA(*)
      DIMENSION IADVCT(*),ITYPE(*)
      DIMENSION UI(*), RINIT(*), RDIM(7,*)
C
      INTEGER ISV
      DOUBLE PRECISION VINIT
      CHARACTER*6 IAM
      PARAMETER(IAM='DMMRXV')

CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC DMMRXV
*     These next lines are only to avoid compiler warnings
      VINIT=0.D0
*
      CALL LOGMES('############# Requesting DMM variables')
C
C
C
C
C     ***********************************************************
C     ***   SET INITIAL VALUES FOR STATE VARIABLES            ***
C     ***                                                     ***
C     ***   The first argument is the user input (previously  ***
C     ***   obtained by reading an input set). The next       ***
C     ***   argument, RINIT, is the list of initial values    ***
C     ***   for each of the internal state variables listed   ***
C     ***   below.                                            ***
C     ***                                                     ***
C     ***********************************************************
C
C
C     For now, the ISVs will be initialized using an assumption
C     that the initial stress is zero.  Host codes that support a
C     nonzero initial stress field will need to call
C     ISOTROPIC_DMMMATERIAL_INIT with the
C     initial stress in each element to reset the
C     ISVs appropriately.
C
      DO ISV=1,NDMMISV
         RINIT(ISV)=PZERO
      ENDDO
C
C
      NX=0
C
C----------------------------------------------------------------------1
      NX=NX+1                                                    !KEQDOT
      IF(NX.NE.KEQDOT)CALL BOMBED(IAM//' KEQDOT pointer wrong')
      KEY(NX) = 'EQDOT'
      NAME(NX) = 'EQDOT'
      IADVCT(NX)=2        ! output only
C     itype=1  (scalar)
      RDIM(1,NX)=  PZERO  ! units: 1/time
      RDIM(2,NX)=  PZERO
      RDIM(3,NX)= -PONE
      RINIT(NX)=VINIT
C----------------------------------------------------------------------2
      NX=NX+1                                                       !kI1
      IF(NX.NE.KI1)CALL BOMBED(IAM//' KI1 pointer wrong')
      NAME(NX)='I1'
C     Note: stress is positive in tension. Therefore, I1 is typically
C     negative in compression. The mechanical pressure (positive in
C     compression) and mean stress (positive in tension) are given by
C           pressure    = - I1/3
C           mean stress = + I1/3
      KEY(NX)='I1'
      IADVCT(NX)=2        ! output only
C     itype=1  (scalar)
      RDIM(1,NX)= -PONE   ! units: stress
      RDIM(2,NX)=  PONE
      RDIM(3,NX)= -PTWO
      RINIT(NX)=VINIT
C----------------------------------------------------------------------3
      NX=NX+1                                                   !KROOTJ2
      IF(NX.NE.KROOTJ2)CALL BOMBED(IAM//' KROOTJ2 pointer wrong')
      NAME(NX)='rootj2'
C     The second stress invariant is given by
C            J2 = (1/2) trace(S.S)
C     where S is the stress deviator. Therefore,
C       ROOTJ2 = magnitude(S)/sqrt(2)
      KEY(NX)='ROOTJ2'
      IADVCT(NX)=2        ! output only
C     itype=1  (scalar)
      RDIM(1,NX)= -PONE   ! units: stress
      RDIM(2,NX)=  PONE
      RDIM(3,NX)= -PTWO
      RINIT(NX)=VINIT
C----------------------------------------------------------------------4
      NX=NX+1                                                     !KEQPS
      IF(NX.NE.KEQPS)CALL BOMBED(IAM//' KEQPS pointer wrong')
      NAME(NX)='EQPS'
C     EQPS is the time integral of
C           SQRT[2.0]*magnitude[deviator[strain rate]
C     It is defined in this way to be conjugate to ROOTJ2.  In other
C     words, the inner product of the stress deviator with the strain
C     rate is equal to ROOTJ2 times the rate of EQPS.
      KEY(NX)='EQPS'
      IADVCT(NX)=0        ! input and output
C     itype=1  (scalar)
C     RDIM=default        ! units:  dimensionless
      RINIT(NX)=VINIT
C----------------------------------------------------------------------5
      NX=NX+1                                                     !KPW
      IF(NX.NE.KPW)CALL BOMBED(IAM//' KPW pointer wrong')
      NAME(NX)='plastic work'
      KEY(NX)='PW'
      IADVCT(NX)=0        ! output only
      ITYPE(NX)=1
C     itype=1  (scalar)
      RINIT(NX)=VINIT
C-----------------------------------------------------------------------6
      NX=NX+1                                                      !KTMPR
      IF(NX.NE.KT)CALL BOMBED(IAM//' KTMPR pointer wrong')
      NAME(NX)='Temperature'
      KEY(NX)='TMPR'
      IADVCT(NX)=1        ! input/output
      RDIM(4,NX)=  PONE
      RINIT(NX)=UI(IPT0)
C-----------------------------------------------------------------------7
      NX=NX+1                                                     !KSNDSP
      IF(NX.NE.KCS)CALL BOMBED(IAM//'1 KSNDSP pointer wrong')
      NAME(NX)='sound speed'
      KEY(NX)='SNDSP'
      IADVCT(NX)=1        ! input/output
      RDIM(1,NX)=  PONE
      RDIM(3,NX)= -PONE
      RINIT(NX)=UI(IPC0)
C-----------------------------------------------------------------------8
      NX=NX+1                                                       !KRHO
      IF(NX.NE.KRHO)CALL BOMBED(IAM//' KRHO pointer wrong')
      NAME(NX)='Density'
      KEY(NX)='DENS'
      IADVCT(NX)=1        ! input/output
      RDIM(1,NX) = -PTHREE
      RDIM(2,NX) =  PONE
      RINIT(NX)=UI(IPRHO0)
C-----------------------------------------------------------------------9
      NX=NX+1                                                     !KENRGY
      IF(NX.NE.KEU)CALL BOMBED(IAM//' KENRGY pointer wrong')
      NAME(NX)='Energy'
      KEY(NX)='ENRGY'
      IADVCT(NX)=1        ! input/output
      RDIM(1,NX) =  PTWO
      RDIM(2,NX) =  PONE
      RDIM(3,NX) = -PTWO
      RINIT(NX)=UI(IPT0)*UI(IPCV)
C----------------------------------------------------------------------10
      NX=NX+1                                                   !Jacobian
      IF(NX.NE.KRJ)CALL BOMBED(IAM//' KRJ pointer wrong')
      NAME(NX)='Jacobian'
      KEY(NX)='JACOBIAN'
      RINIT(NX)=PONE
      IADVCT(NX)=1
C----------------------------------------------------------------------11
      NX=NX+1                                                         !AM
      IF(NX.NE.KAM)CALL BOMBED(IAM//' KAM pointer wrong')
      NAME(NX)='anisotropy'
      KEY(NX)='AM'
      IADVCT(NX)=1
      RINIT(NX)=0.D0
C----------------------------------------------------------------------12
      NX=NX+1                                                       !EQPV
      IF(NX.NE.KEQPV)CALL BOMBED(IAM//' KEQPV pointer wrong')
      NAME(NX)='eqpv'
      KEY(NX)='EQPV'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------13
      NX=NX+1                                                         !F1
      IF(NX.NE.KF1)CALL BOMBED(IAM//' KF1 pointer wrong')
      NAME(NX)='Free 1'
      KEY(NX)='FREE1'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------14
      NX=NX+1                                                         !F2
      IF(NX.NE.KF2)CALL BOMBED(IAM//' KF2 pointer wrong')
      NAME(NX)='Free 2'
      KEY(NX)='FREE2'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------15
      NX=NX+1                                                         !F3
      IF(NX.NE.KF3)CALL BOMBED(IAM//' KF3 pointer wrong')
      NAME(NX)='Free 3'
      KEY(NX)='FREE3'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------16
      NX=NX+1                                                         !F4
      IF(NX.NE.KF4)CALL BOMBED(IAM//' KF4 pointer wrong')
      NAME(NX)='Free 4'
      KEY(NX)='FREE4'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------17
      NX=NX+1                                                         !F5
      IF(NX.NE.KF5)CALL BOMBED(IAM//' KF5 pointer wrong')
      NAME(NX)='Free 5'
      KEY(NX)='FREE5'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------18
      NX=NX+1                                                         !F6
      IF(NX.NE.KF6)CALL BOMBED(IAM//' KF6 pointer wrong')
      NAME(NX)='Free 6'
      KEY(NX)='FREE6'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------19
      NX=NX+1                                                         !F7
      IF(NX.NE.KF7)CALL BOMBED(IAM//' KF7 pointer wrong')
      NAME(NX)='Free 7'
      KEY(NX)='FREE7'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------20
      NX=NX+1                                                         !F8
      IF(NX.NE.KF8)CALL BOMBED(IAM//' KF8 pointer wrong')
      NAME(NX)='Free 8'
      KEY(NX)='FREE8'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------21
      NX=NX+1                                                         !F9
      IF(NX.NE.KF9)CALL BOMBED(IAM//' KF9 pointer wrong')
      NAME(NX)='Free9'
      KEY(NX)='FREE9'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------22
      NX=NX+1                                                        !F10
      IF(NX.NE.KF10)CALL BOMBED(IAM//' KF10 pointer wrong')
      NAME(NX)='Free10'
      KEY(NX)='FREE10'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------23
      NX=NX+1                                                    !KQSSIGXX
      IF(NX.NE.KQSSIGXX)CALL BOMBED(IAM//' KQSSIGXX pointer wrong')
      NAME(NX)='qssig xx'
      KEY(NX)='QSSIGXX'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------24
      NX=NX+1                                                    !KQSSIGYY
      IF(NX.NE.KQSSIGYY)CALL BOMBED(IAM//' KQSSIGYY pointer wrong')
      NAME(NX)='qssig yy'
      KEY(NX)='QSSIGYY'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------25
      NX=NX+1                                                    !KQSSIGZZ
      IF(NX.NE.KQSSIGZZ)CALL BOMBED(IAM//' KQSSIGZZ pointer wrong')
      NAME(NX)='qssig zz'
      KEY(NX)='QSSIGZZ'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------26
      NX=NX+1                                                    !KQSSIGXY
      IF(NX.NE.KQSSIGXY)CALL BOMBED(IAM//' KQSSIGXY pointer wrong')
      NAME(NX)='qssig xy'
      KEY(NX)='EQSSIGXY'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------27
      NX=NX+1                                                   !KQSSIGYZ
      IF(NX.NE.KQSSIGYZ)CALL BOMBED(IAM//' KQSSIGYZ pointer wrong')
      NAME(NX)='qssig yz'
      KEY(NX)='QSSIGYZ'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------28
      NX=NX+1                                                   !KQSSIGZX
      IF(NX.NE.KQSSIGZX)CALL BOMBED(IAM//' KQSSIGZX pointer wrong')
      NAME(NX)='qssig zx'
      KEY(NX)='QSSIGZX'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------29
      NX=NX+1                                                    !KEXX
      IF(NX.NE.KEXX)CALL BOMBED(IAM//' KEXX pointer wrong')
      NAME(NX)='estrain xx'
      KEY(NX)='EXX'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------30
      NX=NX+1                                                    !KEYY
      IF(NX.NE.KEYY)CALL BOMBED(IAM//' KEYY pointer wrong')
      NAME(NX)='estrain yy'
      KEY(NX)='EYY'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------31
      NX=NX+1                                                    !KEZZ
      IF(NX.NE.KEZZ)CALL BOMBED(IAM//' KEZZ pointer wrong')
      NAME(NX)='estrain zz'
      KEY(NX)='EZZ'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------32
      NX=NX+1                                                    !KEXY
      IF(NX.NE.KEXY)CALL BOMBED(IAM//' KEXY pointer wrong')
      NAME(NX)='estrain xy'
      KEY(NX)='EEXY'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------33
      NX=NX+1                                                   !KEYZ
      IF(NX.NE.KEYZ)CALL BOMBED(IAM//' KEYZ pointer wrong')
      NAME(NX)='estrain yz'
      KEY(NX)='EYZ'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------34
      NX=NX+1                                                   !KEZX
      IF(NX.NE.KEZX)CALL BOMBED(IAM//' KEZX pointer wrong')
      NAME(NX)='estrain xz'
      KEY(NX)='EZX'
      IADVCT(NX)=1
      RINIT(NX)=VINIT
C----------------------------------------------------------------------35
      NX=NX+1                                                   !KEJ2
      IF(NX.NE.KEJ2)CALL BOMBED(IAM//' KEJ2 pointer wrong')
      NAME(NX)='estrain'
      KEY(NX)='EDEVJ2'
      IADVCT(NX)=1
      RINIT(NX)=VINIT


C     ##################################################################
      IF(NX.GT.NDMMISV)CALL BOMBED
     & ('INCREASE NDMMISV IN ROUTINE DMMRXV AND IN DATA FILE')
C     convert NAME and KEY to character streams NAMEA and KEYA
C     (See note about TOKENS in prolog of this routine)
      CALL TOKENS(NX,NAME,NAMEA)
      CALL TOKENS(NX,KEY ,KEYA )
C     CALL LOGMES('############# exiting DMMRXV')
      RETURN
      END !SUBROUTINE DMMRXV
