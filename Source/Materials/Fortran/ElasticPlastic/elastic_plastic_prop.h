C***********************************************************************
C     diamm material constants
C     Properties and control parameters, including derived constants
C
C     These include material properties that are treated as if they
C     were constants even if they might have been altered (as from
C     softening or thermal effects) in subroutine DMMVAR_PUT.
C     The true constants are loaded in subroutine DMMPAR_PUT.
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...logicals
      LOGICAL ANISO,WANTSELASTIC,RATEDEP,PRESDEPYLD
      SAVE /DMMPROPL/
      COMMON /DMMPROPL/ANISO,WANTSELASTIC,RATEDEP,PRESDEPYLD
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...integers
C---.----1----.----2----.----3----.----4----.----5----.----6----.----7--
C  ...reals
C     REFR: a characteristic scale for the Lode r coordinate
C     REFZ: a characteristic scale for the Lode z coordinate
      DOUBLE PRECISION B0,B1,B2,G0,G1,G2,G3,TWOG1,A1,A2,A4G,A3,A4,PRAT
      DOUBLE PRECISION T1,T2,T3,T4,RHO0,T0,TM,C0,S1,GP,CV,XP,A5,A6
      SAVE /DMMPROPR/
      COMMON /DMMPROPR/B0,B1,B2,G0,G1,G2,G3,TWOG1,A1,A2,A4G,A3,A4,PRAT,
     $T1,T2,T3,T4,RHO0,T0,TM,C0,S1,GP,CV,XP,A5,A6
