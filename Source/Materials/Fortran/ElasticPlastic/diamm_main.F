c The MIT License
c
c Copyright (c) 2011 Tim Fuller
c
c License for the specific language governing rights and limitations under
c Permission is hereby granted, free of charge, to any person obtaining a
c copy of this software and associated documentation files (the "Software"),
c to deal in the Software without restriction, including without limitation
c the rights to use, copy, modify, merge, publish, distribute, sublicense,
c and/or sell copies of the Software, and to permit persons to whom the
c Software is furnished to do so, subject to the following conditions:
c
c The above copyright notice and this permission notice shall be included
c in all copies or substantial portions of the Software.
c
c THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
c OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
c FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
c THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
c LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
c FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
c DEALINGS IN THE SOFTWARE.

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
#ifdef TJF_IMPLNONE
      IMPLICIT NONE
#else
#include "precision.Blk"
#endif
C
C.............................................................parameters
#include "diamm_pnt.Blk"
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION PTHIRD,PHALF,P3HALF,P2THIRD
      DOUBLE PRECISION PZERO,PONE,PTWO,PTHREE,ROOT2
      DOUBLE PRECISION TOL1M10,YLDTOL
#endif
      PARAMETER (PTHIRD=0.333333333333333333333333333333333333333333D0)
      PARAMETER (P2THIRD=0.66666666666666666666666666666666666666667D0)
      PARAMETER (  ROOT2=0.141421356237309504880168872420969807856967D1)
      PARAMETER (PHALF=0.5D0,P3HALF=1.5D0)
      PARAMETER (PZERO=0.D0,PONE=1.D0,PTWO=2.D0,PTHREE=3.D0)
      PARAMETER (TOL1M10= 1.0D-10,YLDTOL=1.0D-3)
C.................................................................common
#include "diamm_prop.h"
C.................................................................passed
#ifdef TJF_IMPLNONE
      INTEGER NBLK,NINSV
      DOUBLE PRECISION SV,SIG,USM,UI,D,DT
#endif
      DIMENSION SV(NINSV),SIG(6),USM(1)
      DIMENSION UI(*),D(6)
C..................................................................local
#ifdef TJF_IMPLNONE
      INTEGER IJ,I,ndum
      DOUBLE PRECISION DUMA
      DOUBLE PRECISION DUM,RJN
      DOUBLE PRECISION BM,SM,THREEK,TWOG,YF
      DOUBLE PRECISION T,TMLT,U,RHO,DTE,DTP
      DOUBLE PRECISION DELTA,FAC
      DOUBLE PRECISION TRDTAU,SINVDTAU
      DOUBLE PRECISION S,TAUN,QSTAUN,QSTAUT,QSTAUP,TAUP,RI1,RTJ2
      DOUBLE PRECISION QSTAUPISO,QSTAUPDEV,QSTH
      DOUBLE PRECISION DTAUT,DTAU,DTAUISO,DTAUDEV,APART
      DOUBLE PRECISION DE,DEISO,DEDEV,DEP,DEE
      DOUBLE PRECISION EN,EISO,EDEV,DEPDEV,DEPISO,TRDEP,SINVDEP
      DOUBLE PRECISION TRDE,SINVDE,BETA1,BETA2,DEQPS
      DOUBLE PRECISION RNN,RMM,RMAGM,RMAGN,P,Z,TRM,A
      DOUBLE PRECISION RMMI,RMMD,DCSP
      DOUBLE PRECISION DFDI1,DFDJ2,DGDI1,DGDJ2
      DOUBLE PRECISION ALPHA1,ALPHA2,DDD,BETA,Y1,Y2,ZETA
      DOUBLE PRECISION H
#endif
C     symmetric stress tensors
      DIMENSION S(6),TAUN(6),TAUP(6)
      DIMENSION DTAUT(6),DTAU(6),DTAUISO(6),DTAUDEV(6)
      DIMENSION QSTAUPISO(6),QSTAUPDEV(6),QSTAUN(6),QSTAUT(6),QSTAUP(6)
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
#ifdef TJF_IMPLNONE
      DOUBLE PRECISION DMMTR,DMMDBD,DMMEN,DMMYLD
      DOUBLE PRECISION DMMRJ2,DMMTS,DMMTMG
      DOUBLE PRECISION DMMAMA,DMMHRD
#endif
C....................................................statement functions
Ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc DIAMM_CALC
C
      ndum = nblk
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
C     Hardening modulus
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
C     Normally at this point, the following line would occur
      print*,bm,sm,yf,dmmtr(p),dmmrj2(p)
      CALL BOMBED('Newton iterations failed')
C     But, I want the code to be more robust, so I put in a last ditch
C     effort to get back to the yield surface.  Like the kludge above,
C     this was put in specifically for Uintah, where large amount of
C     ringing in shock problems causes everything to wig out.  This just forces
C     the stress deviator to be right on the yield surface.
C
         CALL DMMTD(QSTAUP,QSTAUPISO,QSTAUPDEV,RI1,RTJ2)
         DUM= DMMTMG(QSTAUPDEV)
         DO IJ=1,6
            QSTAUP(IJ)= QSTAUPISO(IJ) + (DUM-ROOT2*YF)/DUM*QSTAUPDEV(IJ)
         ENDDO
         YF= DMMYLD(QSTAUT,SV(KEQPS),T)
         IF(YF.GT.YLDTOL)CALL BOMBED('Newton iterations failed')
         CALL LOGMES('Bad scaling of stress about to occur!')

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
