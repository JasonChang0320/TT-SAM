subroutine input_vel(mod_fn)
    implicit real*8 (a-h,o-z)
    include 'setup.inc'
    character*24 mod_fn
    open(1,file=mod_fn,status='old')
    read(1,*)bld1,bld2,nlon_a,nlat_a,ndep_a
    read(1,*)(lon_a(i),i=1,nlon_a)
    read(1,*)(lat_a(i),i=1,nlat_a)
    read(1,*)(dep_a(i),i=1,ndep_a)
! read P velocity model
    do k=1,ndep_a
       do j=1,nlat_a
          read(1,*)(vp_a(i,j,k),i=1,nlon_a)
       enddo
    enddo
! read S velocity model
    do k=1,ndep_a
       do j=1,nlat_a
          read(1,*)(vs_a(i,j,k),i=1,nlon_a)
       enddo
    enddo
    close(1)
    call bldmap
!   ave=0.0
!   do i=1,nlat_a
!      ave=ave+lat_a(i)
!   enddo
!   ave=ave/real(nlat_a)
!   ro=earthr(ave)
end

subroutine bldmap
    implicit real*8 (a-h,o-z)
    include 'setup.inc'
    real*8 lon_now,lat_now,dep_now

! for local 3D model
    lon1_a=bld1-lon_a(1)
    ilonmax=(1e-10)+(lon_a(nlon_a)+lon1_a)/bld1
    lat1_a=bld1-lat_a(1)
    ilatmax=(1e-10)+(lat_a(nlat_a)+lat1_a)/bld1
    dep1_a=bld2-dep_a(1)
    idepmax=(1e-10)+(dep_a(ndep_a)+dep1_a)/bld2
    if ((ilonmax.gt.ilondeg).or.(ilatmax.gt.ilatdeg).or.(idepmax.gt.idepkm)) then
       print*,"Error, model dimension out of range!"
       stop
    endif
    ilon=1
    do i=1,ilonmax
       ilon1=ilon+1
       lon_now=float(i)*bld1-lon1_a
       if (lon_now.ge.lon_a(ilon1)) ilon=ilon1
       ilonloc_a(i)=ilon
    enddo
    do i=ilonmax+1,ilondeg
       ilonloc_a(i)=0
    enddo
    ilat=1
    do i=1,ilatmax
       ilat1=ilat+1
       lat_now=float(i)*bld1-lat1_a
       if (lat_now.ge.lat_a(ilat1)) ilat=ilat1
       ilatloc_a(i)=ilat
    enddo
    do i=ilatmax+1,ilatdeg
       ilatloc_a(i)=0
    enddo
    idep=1
    do i=1,idepmax
       idep1=idep+1
       dep_now=float(i)*bld2-dep1_a
       if (dep_now.ge.dep_a(idep1)) idep=idep1
       ideploc_a(i)=idep
    enddo
    do i=idepmax+1,idepkm
       ideploc_a(i)=0
    enddo
end

subroutine intmap_3d(lon,lat,dep,ip,jp,kp)
    implicit real*8 (a-h,o-z)
    include 'setup.inc'
    real*8 lon,lat,dep
    ip=int(1e-10+(lon+lon1_a)/bld1)
    jp=int(1e-10+(lat+lat1_a)/bld1)
    kp=int(1e-10+(dep+dep1_a)/bld2)
    if ((ip.le.0).or.(jp.le.0).or.(kp.le.0)) then
       print*,"Error,lon,lat,dep out of range!"
       print*,"lon=",lon,"lat=",lat,"dep=",dep
       print*,"ip,jp,kp",ip,jp,kp
       stop
    endif
    ip=ilonloc_a(ip)
    jp=ilatloc_a(jp)
    kp=ideploc_a(kp)
    if ((ip.eq.0).or.(jp.eq.0).or.(kp.eq.0)) then
       print*,"Error,crust lon,lat out of range!"
       print*,"lon=",lon,"lat=",lat,"dep=",dep
       print*,"ip,jp,kp",ip,jp,kp
       stop
    endif
    return
end

function velocity(r,pa,ra)
    implicit real*8 (a-h,o-z)
    parameter(ro=6371.0)
    real*8 lat,lon,dep,shiftlo
    real*8 r,pa,ra,r2d
    real*8 v,velocity
    common /coord/ shiftlo
    r2d = 90./asin(1.)
    lat=geoc_to_geog(90.0-pa*r2d)
    lon=ra*r2d+shiftlo
    dep=ro-r
    call vel_3d(lon,lat,dep,v)
    velocity=v
    return
end

subroutine vel_3d(lon,lat,dep,v)
    implicit real*8 (a-h,o-z)
    include 'setup.inc'
    real*8 lon,lat,dep,v
    real*8 lonf,lonf1,latf,latf1,depf,depf1
    real*8 wv(2,2,2)
    common /weight/ wv,ip,jp,kp
    if (dep.lt.dep_a(1)) dep=dep_a(1)
    call intmap_3d(lon,lat,dep,ip,jp,kp)
    ip1=ip+1
    jp1=jp+1
    kp1=kp+1
    if ((ip1.gt.nlon_a).or.(jp1.gt.nlat_a).or.(kp1.gt.ndep_a)) then
       print*,"Error, ip1,jp1,kp1 out of range!"
       print*,"ip1=",ip1,"jp1=",jp1,"kp1=",kp1
       stop
    endif
    lonf=(lon-lon_a(ip))/(lon_a(ip1)-lon_a(ip))
    latf=(lat-lat_a(jp))/(lat_a(jp1)-lat_a(jp))
    depf=(dep-dep_a(kp))/(dep_a(kp1)-dep_a(kp))
    lonf1=1.0-lonf
    latf1=1.0-latf
    depf1=1.0-depf
    wv(1,1,1)=lonf1*latf1*depf1
    wv(2,1,1)=lonf*latf1*depf1
    wv(1,2,1)=lonf1*latf*depf1
    wv(2,2,1)=lonf*latf*depf1
    wv(1,1,2)=lonf1*latf1*depf
    wv(2,1,2)=lonf*latf1*depf
    wv(1,2,2)=lonf1*latf*depf
    wv(2,2,2)=lonf*latf*depf
    if (ips.eq.2) then
       v=wv(1,1,1)*vs_a(ip,jp,kp)+wv(2,1,1)*vs_a(ip1,jp,kp)    &
        +wv(1,2,1)*vs_a(ip,jp1,kp)+wv(2,2,1)*vs_a(ip1,jp1,kp)  &
        +wv(1,1,2)*vs_a(ip,jp,kp1)+wv(2,1,2)*vs_a(ip1,jp,kp1)  &
        +wv(1,2,2)*vs_a(ip,jp1,kp1)+wv(2,2,2)*vs_a(ip1,jp1,kp1)
    else
       v=wv(1,1,1)*vp_a(ip,jp,kp)+wv(2,1,1)*vp_a(ip1,jp,kp)    &
        +wv(1,2,1)*vp_a(ip,jp1,kp)+wv(2,2,1)*vp_a(ip1,jp1,kp)  &
        +wv(1,1,2)*vp_a(ip,jp,kp1)+wv(2,1,2)*vp_a(ip1,jp,kp1)  &
        +wv(1,2,2)*vp_a(ip,jp1,kp1)+wv(2,2,2)*vp_a(ip1,jp1,kp1)
    endif
    return
end

real*8 function geog_to_geoc(xla)
    implicit none
    real*8 xla,RAD_PER_DEG,B2A_SQ
    RAD_PER_DEG=0.0174532925199432955
    B2A_SQ=0.993305521
    geog_to_geoc = atan(B2A_SQ*tan(RAD_PER_DEG*xla)) / RAD_PER_DEG
    return
end

real*8 function geoc_to_geog(xla)
    implicit none
    real*8 xla,RAD_PER_DEG,B2A_SQ
    RAD_PER_DEG=0.0174532925199432955
    B2A_SQ=0.993305521
    geoc_to_geog = atan(tan(RAD_PER_DEG*xla)/B2A_SQ) / RAD_PER_DEG
    return
end 

subroutine pbr(evla,evlo,evdp,stla,stlo,stel,w,np,tk)
    implicit real*8(a-h,o-z)
    parameter (msg=16384)
    real*8 w(3,msg+1)
    real*8 r(msg+1), a(msg+1), b(msg+1)
    integer ni,i
    real*8 shiftlo
    real*8 aas,bbs,hs,aar,bbr,hr
    real*8 xfac,flim,mins
    real*8 dpi,r2d
    real*8 velocity,rtim
    real*8 tk
    real*8 as,ar
    real*8 bre,bso,dlo
    real*8 ad,rs,rr
    real*8 x1,y1,z1,x2,y2,z2,x3,y3,z3,dx,dy,dz
    real*8 r1,a1,b1,r2,a2,b2,r3,a3,b3
    real*8 x,y,z,acosa,sina,cosa,to,tp
    real*8 dn,ddn,dr,da,db
    real*8 dseg,ddseg
    real*8 v1,v2,v3
    real*8 upz,dwz
    real*8 vr1,vr2,vr,vb1,vb2,vb,va1,va2,va
    real*8 pr,pa,pb
    real*8 vrd,rvr,rva,rvb,rvs
    real*8 cc,rcur,rdr,rda,rdb,rpr,ap,bp
    real*8 adV,bdV,rdV
    real*8 ro,RNULL
      
    common /coord/ shiftlo
    data     ro  / 6371.00 /
    data    RNULL /0.0e10/

!   right now force the receiver at elevation of 0
!   hr=0.0
!   write(*,*)aas, bbs, hs, aar, bbr, hr
!                                                                       
!   parameters for calculation                                      
!      xfac   = enhancement factor (see um and thurber, 1987)        
!      nloop  = number of bending iterations
!      n1, n2 = min & max of ray segments
!      mins  = min. length of segment (km)
!
!   initialization                                                  

    aas=evla
    bbs=evlo
    hs=evdp
    aar=stla
    bbr=stlo
    hr=stel

    ni     = msg+1
    xfac   = 1.5
    n1     = 2
    n2     = msg
    nloop  = 12800
    flim   = 1.e-4/100.
    mins   = 2.
    dpi = asin(1.)/ 90.
    r2d = 90./asin(1.)

!-- Check coordinates
    if (aas.LT.-90.OR.aas.GT.90.) then
       write(*,*)'Latitude of source is out of range'
       stop
    endif
    if (aar.LT.-90.OR.aar.GT.90.) then
       write(*,*)'Latitude of station is out of range'
       stop
    endif
    if (bbs.LT.-180.OR.bbs.GT.180.) then
       write(*,*)'Longitude of source is out of range'
       stop
    endif
    if (bbr.LT.-180.OR.bbr.GT.180.) then
       write(*,*)'Longitude of station is out of range'
      stop
    endif

!-- longitude and latitude range from 0 to 180. 
!-- This program does not work with angles
!-- greater than 180.       

!-- Pass from latitude to colatitude

    as = (90.00-aas) * dpi
    ar = (90.00-aar) * dpi

    if (bbr.LT.0.0) then
       bre=360.+bbr
    else
       bre=bbr
    endif

    if (bbs.LT.0.0) then
       bso=360.+bbs
    else
       bso=bbs
    endif
    dlo=abs(bso-bre)

    if (dlo.LT.180.) then
       shiftlo=0.0e10
       if (bso.LT.bre) then
          shiftlo=bso-(180.-dlo)/2.
          bbs=(180.-dlo)/2.
          bbr=bbs+dlo
       else
          shiftlo=bre-(180.-dlo)/2.
          bbr=(180.-dlo)/2.
          bbs=bbr+dlo
       endif
    else
       dlo=360.0000-dlo
       shiftlo=0.0e10
       if (bso.LT.bre) then
          shiftlo=bso-(dlo+(180.-dlo)/2.)
          bbs=(180.-dlo)/2.+dlo 
          bbr=bbs-dlo
       else    
          shiftlo=bre-(dlo+(180.-dlo)/2.)
          bbr=(180.-dlo)/2.+dlo
          bbs=bbr-dlo
       endif
    endif

    bs = bbs * dpi
    br = bbr * dpi  
    ad = (as + ar) / 2.                                               
    rs = ro - hs
    rr = ro - hr                                                      
!
! *** initial straight ray ***                                           
!   ni : number of ray segments
    ni = n1
    x1 = rs*sin(as)*cos(bs)                                       
    y1 = rs*sin(as)*sin(bs)                                       
    z1 = rs*cos(as)                                               
    x2 = rr*sin(ar)*cos(br)                                       
    y2 = rr*sin(ar)*sin(br)                                       
    z2 = rr*cos(ar)       
    dx = x2-x1
    dy = y2-y1
    dz = z2-z1
    dlen=sqrt(dx*dx+dy*dy+dz*dz)
    if (ni.lt.2) ni=2
    dx = (x2-x1) / ni
    dy = (y2-y1) / ni
    dz = (z2-z1) / ni
    do j=1,ni+1
       x = x1 + dx*(j-1)
       y = y1 + dy*(j-1)
       z = z1 + dz*(j-1)                                             
       r(j) = sqrt(x**2 + y**2 + z**2)                         
       acosa=z/r(j)
       if (acosa.LT.-1.) acosa=-1.
       if (acosa.GT.1) acosa=1.
       a(j) = acos(acosa)                                   
       acosa=x/r(j)/sin(a(j))
       if (acosa.LT.-1.) acosa=-1.
       if (acosa.GT.1) acosa=1.
       b(j) = acos(acosa)
       if (y.LT.0.00000) b(j)=360.00000*dpi-b(j)
    enddo
    to = rtim(ni+1,r,a,b)
    tp = to
    do i=1,ni+1
       w(1,i) = r(i)
       w(2,i) = a(i)
       w(3,i) = b(i)
    enddo
! *** number of points loop ***
    loops = 0
    do while(ni .le. n2)
! *** interation loop ***                                               
       do l=1,nloop                                                 
          loops = loops + 1
          do kk=2,ni
             !-- see um & thurber (1987) p.974.
             if (mod(kk,2).eq.0) then
                k = kk/2 + 1
             else
                k = ni+1 - (kk-1)/2
             endif
             r1 = r(k-1)                                 
             a1 = a(k-1)                                 
             b1 = b(k-1)                                 
             x1 = r1*sin(a1)*cos(b1)                           
             y1 = r1*sin(a1)*sin(b1)                           
             z1 = r1*cos(a1)                                   
             r3 = r(k+1)                                 
             a3 = a(k+1)                                 
             b3 = b(k+1)                                 
             x3 = r3*sin(a3)*cos(b3)                           
             y3 = r3*sin(a3)*sin(b3)                           
             z3 = r3*cos(a3)                                   
             dx = x3 - x1                                      
             dy = y3 - y1                                      
             dz = z3 - z1                                      
             x2 = x1 + dx/2                                    
             y2 = y1 + dy/2                                    
             z2 = z1 + dz/2                                    
             r2 = sqrt(x2**2 + y2**2 + z2**2)                  
             acosa=z2/r2
             if (acosa.LT.-1.) acosa=-1.
             if (acosa.GT.1) acosa=1.
             a2 = acos(acosa)                                  
             sina = sin(a2)                                    
             cosa = cos(a2)                                    
             acosa=x2/r2/sina
             if (acosa.LT.-1.) acosa=-1.
             if (acosa.GT.1) acosa=1.
             b2 = acos(acosa)                          
             if (y.LT.0.00000) b2=360.00000*dpi-b2
             dn = dx**2 + dy**2 + dz**2                        
             ddn = sqrt(dn)                                    
             dr = (r3-r1) / ddn                                
             da = (a3-a1) / ddn                                
             db = (b3-b1) / ddn
             !-- Begin find the gradients and velocities
             !-- first find the length of segment
             dseg=sqrt((dx/2)**2+(dy/2)**2+(dz/2)**2)
             ddseg=dseg/2.
             !-- Now ddseg will be a distance to find dV along the coordinates 
             !-- Determine velocity at 3 points
             v1 = velocity(r1,a1,b1)                            
             v2 = velocity(r2,a2,b2)                            
             v3 = velocity(r3,a3,b3)                            
             !-- Begin to determine coordinates of points 
             !-- surroundibg point a2,b2,r2 at the distance ddseg
             upz = r2+ddseg
             dwz = r2-ddseg
             if (upz.gt.(ro+10.0)) then  !-- I guess it should be ro+10.0
                upz=ro+10.0
                dwz=upz-dseg
             endif

             if (dwz.le.0.) then
                dwz=0.00000001
                !-- set to ro, mistake?
                upz=ro
             endif

             !-- The following if-endif is just for P & S, thus comment out for SKS & PKP !!!
             !-- This gives the lowermost mantle Vp in the outer core
             vr1 = velocity(upz,a2,b2)
             vr2 = velocity(dwz,a2,b2)
             vr=(vr1-vr2)/dseg
             call km2deg(a2,b2,r2,ddseg,RNULL,adV,bdV,rdV) 
             vb2 = velocity(rdV,adV,bdV)
             call km2deg(a2,b2,r2,-1.*ddseg,RNULL,adV,bdV,rdV)
             vb1 = velocity(rdV,adV,bdV)
             vb=-1.*(vb1-vb2)/dseg
             call km2deg(a2,b2,r2,RNULL,ddseg,adV,bdV,rdV)
             va2 = velocity(rdV,adV,bdV)
             call km2deg(a2,b2,r2,RNULL,-1.*ddseg,adV,bdV,rdV)
             va1 = velocity(rdV,adV,bdV)
             va=-1.*(va1-va2)/dseg

             !-- spherical velocity gradient
             !-- va = va / r2
             !-- vb = vb / r2 / sina
             !-- (tangential vector) = (slowness vector) / s
             pr = dr
             pa = r2 * da
             pb = r2 * sina * db
             vrd = pr*vr + pa*va + pb*vb
             rvr = vr - vrd*pr
             rva = va - vrd*pa
             rvb = vb - vrd*pb
             rvs = sqrt(rvr*rvr + rva*rva + rvb*rvb)               
             if (rvs.eq.0.) then                              
                r(k) = r2                                   
                a(k) = a2                                   
                b(k) = b2                                   
             else                                              
                rvr = rvr / rvs                                 
                rva = rva / rvs                                 
                rvb = rvb / rvs                                 
                cc   = (1./v1+1./v3)/2.                          
                rcur = vr*rvr + va*rva + vb*rvb               
                if (rcur.LE.0.0) then
                   write(*,*)'Negative'
                   rcur=abs(rcur)
                endif
                rcur = (cc*v2+1.) / (4.*cc*rcur)                
                rcur = -rcur + sqrt(rcur**2+dn/(8.*cc*v2))     
                rdr = rvr * rcur
                rda = rva * rcur
                rdb = rvb * rcur
                rpr  = r2 + rdr
                ap  = a2 + rda/r2
                bp  = b2 + rdb/(r2*sina)
                r(k) = (rpr-r(k))*xfac + r(k)
                !-- if r(k)>6371 then force it to the surface.
                if (r(k).gt.(ro+10.0)) r(k)=ro+10.0
                a(k) = (ap-a(k))*xfac + a(k)
                b(k) = (bp-b(k))*xfac + b(k)
             endif
          enddo
          idstn=ni
          do j=1,ni+1
             w(1,j) = r(j)
             w(2,j) = a(j)
             w(3,j) = b(j)
          enddo
          ni=idstn
          tk = rtim(ni+1,r,a,b)
          if (abs(to-tk).le.to*flim) go to 310                        
          to = tk                                                       
       enddo
310    continue   
       to=tk

       !-- skip increasing of segment number if minimum length of segment 
       !-- is exceed or maximum number of segments was reached
       if (dseg.lt.mins.or.ni.ge.n2) then 
          igood=1
          go to 666
       endif

       !-- double the number of points.
       ni = ni * 2
       do i=1,ni/2+1
          r(i*2-1) = w(1,i)
          a(i*2-1) = w(2,i)
          b(i*2-1) = w(3,i)
       enddo
       do k=2,ni,2
          r1 = r(k-1)
          a1 = a(k-1)                                 
          b1 = b(k-1)                                 
          x1 = r1*sin(a1)*cos(b1)                           
          y1 = r1*sin(a1)*sin(b1)                           
          z1 = r1*cos(a1)                                   
          r3 = r(k+1)                                 
          a3 = a(k+1)                                 
          b3 = b(k+1)                                 
          x3 = r3*sin(a3)*cos(b3)                           
          y3 = r3*sin(a3)*sin(b3)                           
          z3 = r3*cos(a3)                                   
          dx = x3 - x1                                      
          dy = y3 - y1                                      
          dz = z3 - z1                                      
          x2 = x1 + dx/2                                    
          y2 = y1 + dy/2                                    
          z2 = z1 + dz/2                                    
          r2 = sqrt(x2**2 + y2**2 + z2**2)                  
          acosa=z2/r2
          if (acosa.LT.-1.) acosa=-1.
          if (acosa.GT.1) acosa=1.
          a2 = acos(acosa)
          sina = sin(a2)                                    
          acosa=x2/r2/sina
          if (acosa.LT.-1.) acosa=-1.
          if (acosa.GT.1) acosa=1.
          b2 = acos(acosa)
          if (y.LT.0.00000) b2=360.00000*dpi-b2
          r(k) = r2
          a(k) = a2
          b(k) = b2
       enddo
       tk = rtim(ni+1,r,a,b)
       !-- here i change tp and put to
       if (abs(to-tk) .le. to*flim) then
          igood=1
          go to 999
       endif
       to = tk 
    enddo                                                                  
999 continue
    !-- write(*,*)"poslednii",ni
    idstn=ni
    do i=1,ni+1
       w(1,i) = r(i)
       w(2,i) = a(i)
       w(3,i) = b(i)
    enddo
    ni=idstn
666 continue

    !-- Return coordinates to the origin
    idstn=ni
    do k=1,ni+1
       w(1,k) = ro-w(1,k)
       w(2,k) = w(2,k)*r2d
       w(2,k) = geoc_to_geog(90.0-w(2,k))
       w(3,k) = w(3,k)*r2d+shiftlo
       if (w(3,k).lt.0.) w(3,k)=360.+w(3,k)
    enddo 
    ni=idstn
    np=ni+1
    !-- convert ray point to cartesion coord.
    return
end
  
subroutine km2deg(ala,alo,adp,dx,dy,bla,blo,bdp)
    implicit real*8(a-h,o-z)
    real*8 ala,alo,adp,dx,dy,bla,blo,bdp
    real*8 dpi,dps
    !-- This subroutine calculate position of new point
    !-- in polar coordinates basing on the coordinates
    !-- of main point in radians ( la is colatitude) and dx and dy in kilometers
    dpi = asin(1.)/ 90.
    dps=adp*SIN(ala)
    blo=alo+atan2(dx,dps)
    bla=ala+atan2(dy,adp)
    if (bla.gt.(180.*dpi)) then
       bla=360.*dpi-bla
       blo=blo+180.*dpi
    endif
    if (bla.lt.0.) then
       bla=abs(bla)
       blo=blo+180.*dpi
    endif
    if (blo.lt.0.) blo=360.*dpi+blo
    if (blo.gt.(360.*dpi)) blo=blo-(360.*dpi)
    bdp=sqrt(adp**2+dx**2+dy**2)
    return
end

function rtim(m, r, a, b)                      
    implicit real*8(a-h,o-z)
    real*8 x1,y1,z1,x2,y2,z2,dl
    real*8 rv2,sm,rv1,rtim
    parameter (msg = 16384)
    real*8 r(msg+1), a(msg+1), b(msg+1)
    integer m
    if (m.GT.(msg+1)) write(*,*)'*'
    rtim = 0.
    rv1 = 1./velocity(r(1),a(1),b(1))
    do j=1,m-1
       x1 = r(j)*sin(a(j))*cos(b(j))
       y1 = r(j)*sin(a(j))*sin(b(j))
       z1 = r(j)*cos(a(j))                                               
       x2 = r(j+1)*sin(a(j+1))*cos(b(j+1))
       y2 = r(j+1)*sin(a(j+1))*sin(b(j+1))
       z2 = r(j+1)*cos(a(j+1))
       dl = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2
       rv2 = 1./velocity(r(j+1),a(j+1),b(j+1))
       sm = (rv1 + rv2) / 2.         
       rtim = rtim + sqrt(dl)*sm
       rv1 = rv2
    enddo                                                        
end
