import numpy as np
import matplotlib.pyplot as plt
from cartopy.mpl import ticker
import cartopy.crs as ccrs

with open("vel3d.mod","r") as file:
    lines = file.readlines()

lon_coor=[]
lon_coor.append([float(x) for x in lines[1].split()])
lon_coor=np.array(lon_coor[0])

lat_coor=[]
lat_coor.append([float(x) for x in lines[2].split()])
lat_coor=np.array(lat_coor[0])

dep_coor=[]
dep_coor.append([float(x) for x in lines[3].split()])
dep_coor=np.array(dep_coor[0])

array=[]
data=lines[4:]
for i,line in enumerate(data):
    # 使用指定分隔符（例如，空格）拆分每一行
    elements = line.split()
    
    # 获取第5列以后的数据
    
    # 添加到结果列表
    array.append([float(x) for x in elements])

array=np.array(array)
reshape_array=array.reshape(2,27,61,76)#(model,depth,lat,lon)


#plot velocity model
X, Y = np.meshgrid(lon_coor, lat_coor)
for model_index,model_name in enumerate(["Vp model","Vs model"]):
    vmax=reshape_array[model_index,:,:].max()
    vmin=reshape_array[model_index,:,:].min()
    for dep_index in range(0,len(dep_coor)):
        fig,ax=plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        ax.coastlines()
        cp = ax.contourf(X, Y, reshape_array[model_index,dep_index,:,:],transform=ccrs.PlateCarree())
        xticks = ticker.LongitudeLocator(nbins=125-119)._raw_ticks(119, 125)
        yticks = ticker.LatitudeLocator(nbins=26-20)._raw_ticks(20, 26)

        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        cbar = fig.colorbar(cp)
        cbar.set_label(f'{model_name[:2]} (km/s)')
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_title(f"Depth: {dep_coor[dep_index]}km")
        fig.savefig(f"model_image/{model_name}_depth_{int(dep_coor[dep_index]*1000)}m.png",dpi=300)

#origin 76 61 27
#after 89 66 27
start=125.08
end=126.5
increased_lon=np.round(np.arange(start, end + 0.08, 0.08),2)
increased_lon_to_str = " ".join("{:.2f}".format(num) for num in increased_lon)

start=26.18
end=26.8
increased_lat=np.round(np.arange(start, end + 0.08, 0.08),2)
increased_lat_to_str = "  ".join("{:.2f}".format(num) for num in increased_lat)


with open("vel3d_new.mod","w") as file:
    points_info=lines[0]
    updated_points_info=points_info.replace("76",f"{76+len(increased_lon)}").replace("61",f"{61+len(increased_lat)}")
    file.write(updated_points_info)
    updated_lon=lines[1].rstrip("\n")+" "+increased_lon_to_str+"\n"
    file.write(updated_lon)
    updated_lat=lines[2].rstrip("\n")+"  "+increased_lat_to_str+"\n"
    file.write(updated_lat)
    file.write(lines[3])
    new_lines=[]
    for i in range(4,len(lines)):

        formatted_number= "{:.3f}".format(array[i-4][-1])
        result_string=("   "+str(formatted_number))*len(increased_lon)
        new_line=lines[i].rstrip("\n")+result_string+"\n"

        if (i-4)%61==60:
            for j in range(len(increased_lat)):
                file.write(new_line)
        file.write(new_line)

#記得手動再增加一個深度

