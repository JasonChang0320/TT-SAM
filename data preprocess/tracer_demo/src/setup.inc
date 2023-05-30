!------ DIMENSION SETTING ------
! Data
    integer maxnsta
    parameter(maxnsta=1500)
! Model     
    integer maxnlat,maxnlon,maxndep,ilatdeg,ilondeg,idepkm
    parameter(maxnlat=100)
    parameter(maxnlon=100)
    parameter(maxndep=100)
    !-ilatdeg, ilondeg are size of map in latitude, longitude
    parameter(ilatdeg=100000)
    parameter(ilondeg=100000)
    parameter(idepkm=100000)

!------ PARAMETER SETTING ------
! getdata
    real*8 sta_loc(maxnsta,3),ray_tt(maxnsta,2),ray_wei(maxnsta,2)
    real*8 olon,olat,odep,omag
    integer date,time,sta_idx(maxnsta),npair
    character*6 sta_nm(maxnsta)
    character*12 pfile
    common/data/sta_loc,ray_tt,ray_wei,date,time,olon,olat,odep,omag,sta_idx,npair,sta_nm,pfile
! input_vel
    real*8 bld1,bld2
    real*8 lat_a(maxnlat),lon_a(maxnlon),dep_a(maxndep)
    real*8 vp_a(maxnlon,maxnlat,maxndep),vs_a(maxnlon,maxnlat,maxndep)
    integer nlat_a,nlon_a,ndep_a,ips
    common/vmodel_a/vp_a,vs_a,lat_a,lon_a,dep_a,bld1,bld2,nlat_a,nlon_a,ndep_a,ips
! bldmap
    real*8 lat1_a,lon1_a,dep1_a
    integer ilonloc_a(ilondeg),ilatloc_a(ilatdeg),ideploc_a(idepkm)
    common/loc_a/ lat1_a,lon1_a,dep1_a,ilonloc_a,ilatloc_a,ideploc_a
