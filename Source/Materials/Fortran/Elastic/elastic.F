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

! This file contains subroutines for Hooke's law
! HOOKECHK: Checks validity of user input and sets defaults
! HOOKERXV: Defines internal state variable requirements
! HOOKE: Main physics driver
! SEZME: pressure-density isentropic EOS for the fluid
!
C
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7-|--.----
      SUBROUTINE HOOKE_INCREMENTAL(NBLK,NINSV,DT,PROP,SIGARG,D,SVARG
     $     ,USM)
C PUBLIC: called by host code
C         This is the main physics routine, called every timestep for
C         every Hooke's Law element that is to be evaluated in rate form.
!
!***********************************************************************
!
!     DESCRIPTION: HOOKE'S LAW IN INCREMENTAL FORM.
!
!     FORMAL PARAMETERS:
!i      NBLK   INTEGER   NUMBER OF ELEMENTS IN THIS BLOCK
!                        Parallel codes normally send NBLK=1
!                        Vectorized codes send more than one element
!                        at a time (NBLK>1).
!i      NINSV  INTEGER   NUMBER OF INTERNAL STATE VARIABLES
!i      DT     REAL      TIME STEP
!i      PROP   REAL      MATERIAL PROPERTIES (see below)
!io     SIGARG REAL      STRESSES
!                        On input, send stress at beginning of step
!                        Output is stress at end of step
!i      D      REAL      STRAIN RATES (preferably at the half step)
!io     SVARG  REAL      INTERNAL STATE VARIABLES (see below)
!o      USM    REAL      Uniaxial strain modulus (with which the host code
!                        may compute an UPPER BOUND on the longitudinal
!                        wave speed by dividing USM by density and taking
!                        the square root of the result)
!
!     CALLED BY:host code during physics loop over elements/cells
!
!     REF: Any undergraduate mechanics of materials book.
!
!***********************************************************************
!
#ifdef SIERRA_PARALLEL_MPI
#include <src/material/Smod_precision.blk>
#else
#include "precision.Blk"
#endif
!
!***********************************************************************
!
! Material Property and control Parameters
!
!
!***********************************************************************
!
!  State variables
!
!***********************************************************************
!
C...parameters
C...parameters (numbers)
      PARAMETER (PTHIRD=0.3333333333333333333333333333333333333333333D0)
      PARAMETER (PZERO=0.0D0,PONE=0.1D1,PTWO=0.2d1)
#include "elastic.pnt"
C...passed
      DIMENSION SVARG(NINSV,NBLK),SIGARG(6,NBLK),USM(NBLK)
      DIMENSION PROP(*),D(6,NBLK)
C...local
      character*18 iam
      parameter (iam='HOOKE_INCREMENTAL')
C
#ifdef BRANNON_DEBUG
#include "debug.h"
#endif
C
! Statement functions
C Specialized version of exponential, apparently implemented to prevent
C underflow replacement of EXP(Z) by zero when Z is an extraordinarily
C large negative number. For all practical purposes, EXPS should be
C regarded as equivalent to EXP.
      PARAMETER (EUNDERFLOW=-34.53877639491D0*PONE)
      PARAMETER (EOVERFLOW=92.1034037D0*PONE)
      EXPS(ARG) = EXP(MIN(MAX(ARG,EUNDERFLOW),EOVERFLOW))
Cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     avoid compiler warning of unused variables
      dum=svarg(1,1)
c
      IBLK=1
      dev=(d(1,iblk)+d(2,iblk)+d(3,iblk))*dt
      twog=ptwo*prop(ipg)
      rlam=prop(iplam)
      do i=1,6
        sigarg(i,iblk)=sigarg(i,iblk)+twog*d(i,iblk)*dt
      enddo
      do i=1,3
        sigarg(i,iblk)=sigarg(i,iblk)+rlam*dev
      enddo
      USM(IBLK)=prop(ipH)
C
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE HOOKECHK ( UI, GC, DC)
C
C PUBLIC: called by host code after user inputs have been obtained from
C the user and assembled into the array UI (called PROP in the main
C physics routine).
C***********************************************************************
C     REQUIRED MIG DATA CHECK ROUTINE
C     Checks validity of user inputs for Hooke's Law model.
C     Sets defaults for unspecified user input.
C     Adjusts user input to be self-consistent.
C
C     input
C     -----
C        UI: user input as read and stored by host code.
C
C        Upon entry, the UI array contains the user inputs EXACTLY
C        as read from the user. These inputs must be ordered in the
C        UI array as follows:
C
C
C         DC: Not used with this model
C
C     Other output
C     ------------
C         GC: Not used with this model
C         DC: Not used with this model
C         Because GC and DC are not used, you may call this routine
C         with a line of the form “CALL HOOKECHK(UI,UI,UI)”
C
C***********************************************************************
C  author: Rebecca Brannon
C
C  yymmdd:usernam: m o d i f i c a t i o n
C  ---------------------------------------------------------------------
C  111111:rmbrann:Created original data check
#ifdef SIERRA_PARALLEL_MPI
#include <src/material/Smod_precision.blk>
#include <src/material/Smod_numbers.blk>
#else
#include "precision.Blk"
#include "numbers.Blk"
#endif
#include "elastic.pnt"
C ...local
      CHARACTER*8 IAM
      PARAMETER( IAM = 'HOOKECHK' )
CCCC  character*60 jnkstr
      LOGICAL DEJAVU
      DATA DEJAVU/.FALSE./
C
C
      DIMENSION UI(*), GC(*), DC(*)
c....................................................statement functions
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c     avoid compiler warning of unused variables
      dum=gc(1)
      dum=dc(1)
c
      if(dejavu)return
C
c     DEJAVU=(NINT(UI(IPGMM+IPDEJAVU)).NE.0)
C     The logical DEJAVU is true if Kayenta parameters have been
C     checked. By extension, if they have been checked, then so have
C     the effective stress parameters, and there is no need to remain.
      IF(.NOT.DEJAVU)THEN
         CALL LOGMES('############# Hookes Law version 000000')
      ELSE
         RETURN
      ENDIF
C
!     call elaspar(rho,0,0,ui(iplam))
C
C     Derived constants
      CALL ELASPAR(UI(IPRHO),0,0,UI(1))
      RETURN
      END
C
C
C
C
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE HOOKERXV(UI,GC,DC,
     & NX, NAMEA, KEYA, RINIT, RDIM, IADVCT, ITYPE)
C PUBLIC: called by host code
C For overall meanings of the calling arguments, see the prolog of
C subroutine KMMRXV. This routine does the same thing as that one, but
C prepends the ISV array with state variables needed for effective stress
C modeling, and it also changes the input/output specification for a
C couple of Kayenta state variables.
C***************************************************************STEP E
C
C  author: Rebecca Brannon
C
C    who yymmdd M O D I F I C A T I O N
C  ------- ------ ----------------------------------------------------
C  rmbrann 111111 Created original extra variable routine
C
#include "precision.Blk"
#include "numbers.Blk"
#include "elastic.pnt"
C
      parameter (nhookeisv=2)
      INTEGER MMCN,MMCK,MNUNIT,MMCNA,MMCKA
      PARAMETER (MMCN=40,MMCK=10,MNUNIT=7)
      PARAMETER (MMCNA=NHOOKEISV*MMCN,MMCKA=NHOOKEISV*MMCK)
C
      CHARACTER*(MMCN) NAME(NHOOKEISV)
      CHARACTER*(MMCK) KEY(NHOOKEISV)
      CHARACTER*1 NAMEA(*), KEYA(*)
      DIMENSION IADVCT(nhookeisv),ITYPE(nhookeisv)
      DIMENSION UI(*), GC(*), DC(*), RINIT(nhookeisv), RDIM(7,nhookeisv)
C
      CHARACTER*8 IAM
      PARAMETER(IAM='HOOKERXV')
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c
      dum=ui(1)
      dum=gc(1)
      dum=dc(1)
      dum=rdim(1,1)
      CALL LOGMES('############# Requesting Hookes Law ISVs')
C
      NX=0
C
C----------------------------------------------------------------------1
      NX=NX+1                                                     ! TEST
      IF(NX.NE.KFREE01)CALL BOMBED(IAM//' KFREE01 pointer wrong')
      NAME(NX)='Free isv 1'
      KEY(NX)='FREE1'
      IADVCT(NX)=1        ! input and output
      ITYPE(NX)=1  !(scalar)
      RINIT(NX)= 0.0
C      RDIM= default   ! units: dimensionless
C
      NX=NX+1                                                     ! TEST
      IF(NX.NE.KFREE02)CALL BOMBED(IAM//' KFREE02 pointer wrong')
      NAME(NX)='Free isv 2'
      KEY(NX)='FREE2'
      IADVCT(NX)=1        ! input and output
      ITYPE(NX)=1  !(scalar)
      RINIT(NX)= 1.0
C      RDIM= default   ! units: dimensionless
C
C ##################################################################
      IF(NX.GT.NHOOKEISV)CALL BOMBED
     & ('INCREASE nHOOKEisv IN ROUTINE HOOKERXV AND IN DATA FILE')
C     convert NAME and KEY to character streams NAMEA and KEYA
C     (See note about TOKENS in prolog of this routine)
      CALL TOKENS(NX,NAME,NAMEA)
      CALL TOKENS(NX,KEY ,KEYA )
C CALL LOGMES(‘############# exiting HOOKERXV’)
      RETURN
      END
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
C
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
      SUBROUTINE ELASPAR(RHO, IP1, IP2, CNST)
C***********************************************************************
C     PURPOSE: This routine computes all elastic constants from any
C     two independent ones. The parameters are...
C
C cnst(1) = lam = First Lame parameter         = G(E-2G)/(3G-E) = K-2G/3
C cnst(2) =   G = Shear mod (2nd Lame parameter, mu)   = E/2/(1+nu)
C cnst(3) =   E = Young's modulus              = 3K(1-2nu) = 2G(1+nu) = 9KG/(3K+
C cnst(4) = rnu = Poisson's ratio              = (3K-E)/6K = lam/2/(lam+G)
C cnst(5) =   K = bulk modulus                 = E/3/(1-2nu) = lam + 2G/3
C cnst(6) =   H = constrained modulus          = 2G+lam = 3K-2lam = K + 4G/3
C cnst(7) =  ko = SIGy/SIGx in uniaxial strain = nu/(1-nu)
C cnst(8) =  cl = longitudinal wave speed      = sqrt(H/rho)
C cnst(9) =  ct = shear (TRANSVERSE) wave speed  = sqrt(G/rho)
C cnst(10)=  co = bulk/plastic wave speed      = sqrt(K/rho)=SQRT(cl^2-4(ct^2)/3
C cnst(11)=  cr = thin rod elastic wave speed  = sqrt(E/rho)
C
C input
C -----
C   RHO: mass density (not used if no wave speeds specified)
C   IP1, IP2: integer ID of the two parameters that are specified
C           (if IP1=IP2=0, then this routine will check if there are
C            two nonzero ones and set IP1 and IP2 to those values. If there
C            is only one nonzero one, then the other significant one
C            will be assumed to be nu=0.0)
C
C input and output
C ----------------
C   PAR: elastic parameters
C
C
C  AUTHORS
C  rmb:Rebecca Brannon:theory, algorithm, and code
C
C  MODIFICATION HISTORY
C  yymmdd|who|what was done
C  ------ --- -------------
C  080613|rmb|created routine
C
C
#ifdef BRANNON_IMPLNONE
      IMPLICIT NONE
#else
#include "implicit.h"
#endif
C
C.............................................................parameters
#ifdef BRANNON_IMPLNONE
      INTEGER IPLAM,IPG,IPE,IPNU,IPK,IPH,IPKO,IPCL,IPCT,IPCO,IPCR
      INTEGER IPNEEDRHO,IPLAST
#endif
      PARAMETER (IPLAM=1)
      PARAMETER (IPG  =2)
      PARAMETER (IPE  =3)
      PARAMETER (IPNU =4)
      PARAMETER (IPK  =5)
      PARAMETER (IPH  =6)
      PARAMETER (IPKO =7)
      PARAMETER (IPNEEDRHO=8)
      PARAMETER (IPCL =8)
      PARAMETER (IPCT =9)
      PARAMETER (IPCO =10)
      PARAMETER (IPCR =11)
      PARAMETER (IPLAST =11)
      CHARACTER*7 IAM
      PARAMETER (IAM='ELASPAR')
      DOUBLE PRECISION
     &     R0,R1,R2,R3,R4,R5,R6,R8,R9,R10,R15,PUNY,R32,R45,R16
      PARAMETER (R0=0.0D0,R1=0.1D1,R2=0.2D1,R3=0.3D1,R4=0.4D1,
     &           R5=0.5D1,R6=0.6D1,R8=0.8D1,R9=0.9D1,R10=0.1D2,
     &           R15=0.15D2,PUNY=0.1D-99,R32=0.32D2,R45=0.45D2,
     &           R16=0.16D2)
C.................................................................common
C.................................................................passed
      INTEGER IP1,IP2
      DOUBLE PRECISION CNST,RHO
      DIMENSION CNST(IPLAST)
C..............................................................functions
C...............................................................external
C..........................................................local scalars
      INTEGER J1,J2,ICASE,IP
      DOUBLE PRECISION RLAM,G,E,RNU,RK,H,RKO,CL,CT,CO,CR
      DOUBLE PRECISION R2G,OPN,OMN,OM2N,A,B
C...........................................................local arrays
C..................................................................saved
C...................................................................data
C....................................................statement functions
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
      J1=IP1
      J2=IP2
C
C
C     If j1 and j2 are zero, scan the list to see which TWO params are
C     non-zero. More than two will cause a bomb.
      IF(J1.LE.0  .AND.  J2.LE.0)THEN
        J1=0
        J2=0
        DO IP=1,IPLAST
          IF(CNST(IP).NE.R0)THEN
            IF(J1.EQ.0)THEN
                J1=IP
            ELSEIF(J2.EQ.0)THEN
                J2=IP
            ELSE
               GO TO 785
                CALL BOMBED
     $          ('too many elastic constants sent to ELASPAR')
            ENDIF
          ENDIF
        ENDDO
      ENDIF
 785  CONTINUE
      IF(J1.EQ.0)CALL BOMBED('no elastic constants sent to ELASPAR')
C
C     If only one nonzero param was found, assume that the other one
C     is zero Poisson's ratio.
      IF(J2.EQ.0)J2=IPNU
C
      IF(J1.EQ.J2)CALL BOMBED(IAM//': not enough elastic constants')
C
C
C
        RLAM = CNST(IPLAM)
           G = CNST(IPG)
           E = CNST(IPE)
         RNU = CNST(IPNU)
          RK = CNST(IPK)
           H = CNST(IPH)
         RKO = CNST(IPKO)
          CL = CNST(IPCL)
          CT = CNST(IPCT)
          CO = CNST(IPCO)
          CR = CNST(IPCR)
C
      IF(RHO.LE.R0.AND.(J1.GE.IPNEEDRHO.OR.J2.GE.IPNEEDRHO))THEN
         CALL BOMBED
     $   (IAM//': Density must be positive when a wave speed is given')
      END IF
C
C
      IF (J1.EQ.IPKO)THEN
         J1=IPNU
         CNST(IPNU)=RKO/(R1+RKO)
      ELSE IF(J1.EQ.IPCL)THEN
         J1=IPH
         CNST(IPH)=RHO*CL*CL
      ELSE IF(J1.EQ.IPCT)THEN
         J1=IPG
         CNST(IPG)=RHO*CT*CT
      ELSE IF(J1.EQ.IPCO)THEN
         J1=IPK
         CNST(IPK)=RHO*CO*CO
      ELSE IF(J1.EQ.IPCR)THEN
         J1=IPE
         CNST(IPE)=RHO*CR*CR
      END IF
C
      IF (J2.EQ.IPKO)THEN
         J2=IPNU
         CNST(IPNU)=RKO/(R1+RKO)
      ELSE IF(J2.EQ.IPCL)THEN
         J2=IPH
         CNST(IPH)=RHO*CL*CL
      ELSE IF(J2.EQ.IPCT)THEN
         J2=IPG
         CNST(IPG)=RHO*CT*CT
      ELSE IF(J2.EQ.IPCO)THEN
         J2=IPK
         CNST(IPK)=RHO*CO*CO
      ELSE IF(J2.EQ.IPCR)THEN
         J2=IPE
         CNST(IPE)=RHO*CR*CR
      END IF
C
C
        RLAM = CNST(IPLAM)
           G = CNST(IPG)
           E = CNST(IPE)
         RNU = CNST(IPNU)
          RK = CNST(IPK)
           H = CNST(IPH)
C
C
C At this point, J1 and J2 each range from 1 to 6, and are distinct.
C There are 15 possible ways to choose 2 numbers from 6:
C
C
      icase = 0
      IF(J1.LT.J2)THEN
         ICASE=10*J1+J2
      ELSE IF(J1.GT.J2)THEN
         ICASE=10*J2+J1
      ENDIF
C      PRINT*,'j1,j2,icase=',J1,J2,ICASE
C
C     Get G and rnu
      IF(ICASE.EQ.12)THEN                                       !rlam, G
         IF(RLAM+G.EQ.R0)GO TO 91
         RNU = RLAM/R2/(RLAM+G)
      ELSE IF(ICASE.EQ.13)THEN                                  !rlam, E
         A  = E*E + R2*RLAM*E + R9*RLAM*RLAM
         IF(A.LT.R0)GO TO 91
         A = SQRT(A)
         G  = (A - R3*RLAM+E)/R4
         RNU = (A-E-RLAM)/R4/RLAM
      ELSE IF(ICASE.EQ.14)THEN                                  !rlam,rnu
         IF(RNU.EQ.R0)GO TO 92
         G  = RLAM*(R1-R2*RNU)/R2/RNU
      ELSE IF(ICASE.EQ.15)THEN                                  !rlam,rK
         IF(R3*RK-RLAM.EQ.R0)GO TO 91
         G  = R3*(RK-RLAM)/R2
         RNU = RLAM/(R3*RK-RLAM)
      ELSE IF(ICASE.EQ.16)THEN                                  !rlam, H
         IF(H+RLAM.EQ.R0)GO TO 91
         G  = (H-RLAM)/R2
         RNU = RLAM/(H+RLAM)
      ELSE IF(ICASE.EQ.23)THEN                                  !   G, E
         RNU = (E-R2*G)/R2/G
      ELSE IF(ICASE.EQ.24)THEN                                  !   G,rnu
         CONTINUE
      ELSE IF(ICASE.EQ.25)THEN                                  !   G,rK
         IF(R3*RK+G.EQ.R0)GO TO 91
         RNU = (R3*RK-R2*G)/R2/(R3*RK+G)
      ELSE IF(ICASE.EQ.26)THEN                                  !   G, H
C         PRINT*,'H,G,H-G=',H,G,H-G
         IF(H-G.EQ.R0)GO TO 91
         RNU = (H-R2*G)/R2/(H-G)
C         PRINT*,'rnu=',RNU
      ELSE IF(ICASE.EQ.34)THEN                                  !   E,rnu
         IF(R1+RNU.EQ.R0)GO TO 91
         G  = E/R2/(R1+RNU)
      ELSE IF(ICASE.EQ.35)THEN                                  !   E,rK
         IF(R9*RK-E.EQ.R0 .OR. RK.EQ.R0)GO TO 91
         G  = R3*E*RK/(R9*RK-E)
         RNU = (R3*RK-E)/R6/RK
      ELSE IF(ICASE.EQ.36)THEN                                  !   E, H
         B = E*E+R9*H*H-R10*E*H
         IF(B.LT.R0 .OR. H.EQ.R0)GO TO 91
         B = SQRT(B)
         G  = (R3*H-B+E)/R8
         RNU = (B-H+E)/R4/H
      ELSE IF(ICASE.EQ.45)THEN                                  ! rnu,rK
         IF(R1+RNU.EQ.R0)GO TO 91
         G  = R3*RK*(R1-R2*RNU)/R2/(R1+RNU)
      ELSE IF(ICASE.EQ.46)THEN                                  ! rnu, H
         IF(R1-RNU.EQ.R0)GO TO 91
         G  = H*(R1-R2*RNU)/R2/(R1-RNU)
      ELSE IF(ICASE.EQ.56)THEN                                  !  rK, H
         IF(R3*RK+H.EQ.R0)GO TO 91
         G  = R3*(H-RK)/R4
         RNU = (R3*RK-H)/(R3*RK+H)
      ELSE
         CALL BOMBED(IAM//': unexpected icase')
      END IF
C
C
      OM2N = R1-R2*RNU
      OMN  = R1-RNU
      OPN  = R1+RNU
      R2G = R2*G
C
      RLAM = R2G*RNU/OM2N
      E    = R2G*OPN
      RK   = R2G*OPN/R3/OM2N
      H    = R2G*OMN/OM2N
      RKO   = RNU/OMN
C
C      PRINT*,'icase=',ICASE
C      PRINT*,'G,rk=',G,RK
      IF(G.LE.R0  .OR.  RK.LE.R0)GO TO 91
C
      CNST(1) = RLAM
      CNST(2) = G
      CNST(3) = E
      CNST(4) = RNU
      CNST(5) = RK
      CNST(6) = H
      CNST(7) = RKO
C
      IF(RHO.NE.R0)THEN
        CNST(8) = SQRT(H/RHO)
        CNST(9) = SQRT(G/RHO)
        CNST(10) = SQRT(RK/RHO)
        CNST(11) = SQRT(E/RHO)
      END IF
      RETURN
   91 CALL BOMBED(IAM//': nonpositive definite elastic constants')
   92 CALL BOMBED(IAM//': ambiguous elastic params')
      RETURN
      END
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
