! This is a script to calculate the travel time between two points in 3D velocity model
program tracer
    implicit real*8 (a-h,o-z)
    include 'setup.inc'
    integer msg,imode,ibin,ipath
    parameter (msg=16384)
    real*8 w(3,msg+1),tp,ts
    real*8 evlo,evla,evla2,evdp
    real*8 stlo,stla,stla2,stel
    character*32 evt_fn,sta_fn,mod_fn

! default file names of 3D velocity model and output travel time table
    mod_fn='vel3d.mod'

! read 3D velocity model
    print*,'read default velocity model [vel3d.mod]'
    call input_vel(mod_fn)

666 print*,'Mode option: (1) input two points'
    print*,'             (2) input two files'
    read(*,*)imode

! two-point mode
    if (imode.eq.1) then
       print*,'Input point 1 (lon, lat, dep):'
       read(*,*)evlo,evla,evdp
       evla2=geog_to_geoc(evla)
       print*,'Input point 2 (lon, lat, dep):'
       read(*,*)stlo,stla,stel
       stla2=geog_to_geoc(stla)

       print*,'Set 1 to output ray path, otherwise 0:'
       read(*,*)ipath

       !-P-wave
       ips=1
       call pbr(evla2,evlo,evdp,stla2,stlo,stel,w,np,tp)
       if (ipath.eq.1) then
          open(11,file='P_path.txt',status='unknown')
          do i=1,np
             write(11,'(3f10.3)')w(3,i),w(2,i),w(1,i)
          enddo
          close(11)
       endif
       !-S-wave
       ips=2
       call pbr(evla2,evlo,evdp,stla2,stlo,stel,w,np,ts)
       if (ipath.eq.1) then
          open(11,file='S_path.txt',status='unknown')
          do i=1,np
             write(11,'(3f10.3)')w(3,i),w(2,i),w(1,i)
          enddo
          close(11)
       endif
       print*,'P- and S-wave ray path were outputed!'
       print*,''

       print*,'--------------------------------------------'
       print*,'P-wave travel time (sec):',tp
       print*,'S-wave travel time (sec):',ts


! two-file mode
    elseif (imode.eq.2) then
       print*,'Input source file:'
       read(*,*)evt_fn
       print*,'Input receiver file:'
       read(*,*)sta_fn

       print*,'Set 1 to output ray path, otherwise 0:'
       read(*,*)ipath

777    print*,'Format of output table: 1) ascii, 2) binary'
       read(*,*)ibin

       open(1,file=evt_fn,status='old')
       open(2,file=sta_fn,status='old')
       if (ibin.eq.1) then
          open(3,file='tt.table',status='unknown')
       elseif (ibin.eq.2) then
          open(3,file='tt.bin',status='unknown',form='unformatted')
       else
          print*,'Option can not be recognized! Please re-try!'
          goto 777
       endif
       ibyte=0
       if (ipath.eq.1) then
          open(10,file='P_path.txt',status='unknown')
          open(11,file='S_path.txt',status='unknown')
       endif
       do
          read(1,*,iostat=ierr1)evlo,evla,evdp
          if (ierr1.lt.0) exit
          evla2=geog_to_geoc(evla)
          do
             read(2,*,iostat=ierr2)stlo,stla,stel
             if (ierr2.lt.0) exit
             stla2=geog_to_geoc(stla)
             stel=-stel/1000.

             !-P-wave
             ips=1
             call pbr(evla2,evlo,evdp,stla2,stlo,stel,w,np,tp)
             !-output P-wave path
             if (ipath.eq.1) then
                do i=1,np
                   write(10,'(3f10.3)')w(3,i),w(2,i),w(1,i)
                enddo
                write(10,'(a1)')"X"
             endif
             !-S-wave
             ips=2
             call pbr(evla2,evlo,evdp,stla2,stlo,stel,w,np,ts)
             !-output S-wave path
             if (ipath.eq.1) then
                do i=1,np
                   write(11,'(3f10.3)')w(3,i),w(2,i),w(1,i)
                enddo
                write(11,'(a1)')"X"
             endif
            
             !-output calculated travel time table
             if (ibin.eq.1) write(3,'(2(2f8.3,f7.3,1x),2f10.3)')evlo,evla,evdp,stlo,stla,stel,tp,ts
             if (ibin.eq.2) write(3)evlo,evla,evdp,stlo,stla,stel,tp,ts
          enddo
          rewind(2)
       enddo
       close(10)
       close(11)
       print*,'--------------------------------------------'
       if (ibin.eq.1) print*,'Ascii table [tt.table] completed!'
       if (ibin.eq.2) print*,'Binary table [tt.bin] completed!'

    else
       print*,'Option can not be recognized! Please re-try!'
       goto 666
    endif
    close(1)
    close(2)
    close(3)
end
