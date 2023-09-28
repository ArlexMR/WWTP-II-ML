import datetime
mgd_inflow       = {'MF' : 'MFWQTC(MGD)',
                    'DR' : 'DRGWQTC(MGD)',
                    'CC' : 'CCWQTC(MGD)', 
                    'FF' : 'FFWQTP(MGD)',
                    'HC' : 'HCWQTP(MGD)'
                    }

conc             = {'MF' : 'MFWQTC',
                    'DR' : 'DRGWQTC ',
                    'CC' : 'CCWQTC ', 
                    'FF' : 'FFWQTC',
                    'HC' : 'HCWQTC'
                    }
titles           = {'MF' : 'A) Morris Forman',
                    'DR' : "B) Derek R. Guthrie",
                    'CC' : 'C) Cedar Creek', 
                    'FF' : 'D) Floyds Fork',
                    'HC' : 'E) Hite Creek'}

gauges           = {'MF' : ['03293510', '03293000'],
                    'DR' : ['03293000', '03292555'],
                    'CC' : ['03293000', '03292500'], 
                    'FF' : ['03293000', '03292555'],
                    'HC' : ['03293530', '03293510']
                    }

gauge_names = {"03293000": "Gage #3",
                "03293500": "03293500",          
                "03292500": "Gage #1",           
                "03292555": "Gage #2",       
                "03293530": "Gage #4",        
                "03293510": "Gage #5"}
def plot_inflow_PMMoV_time_series(msd_mgd, df_PMMoV, wwtp_id, ax):
    mgd2cms = 0.043812636574074
    ax2 = ax.twinx()

    # ax.plot(usgs_mgd[gauges[wwtp_id]], label=gauges[wwtp_id])
    inflow_line = ax.plot(msd_mgd[mgd_inflow[wwtp_id]]*mgd2cms,'k' ,label = 'Inflow')
    PMMoV_line  = ax2.plot(df_PMMoV[conc[wwtp_id]]/1e6,'.k', label = 'PMMoV')

    ax.set_xlim([datetime.date(2020, 8, 1), datetime.date(2021, 8, 1)])

    if wwtp_id == "CC":
        ax.set_ylabel('Influent Rate ($m^3$ $s^{-1}$)', fontsize = 12)
        ax2.set_ylabel(r'PMMoV Avg. C/ml ($\times 10^6$)', fontsize = 12)

    ax.set_title(titles[wwtp_id], loc = 'left', fontsize = 14)

    return inflow_line, PMMoV_line

def plot_scatter_plots(x, y, wwtp_id, ax):
    mgd2cms = 0.043812636574074
    ax.scatter(x*mgd2cms,y/1e6, color = 'k')
    ax.set_xlabel(gauge_names[x.name])

    if wwtp_id == 'MF':
        ax.set_ylabel(r'PMMoV Avg. C/ml ($\times 10^6$)', fontsize =12)
    ax.set_title(titles[wwtp_id], loc = 'left',fontsize = 14)
    ax.tick_params(labelsize = 9)
    if wwtp_id == "CC":
        ax.annotate("Discharge ($m^3$ $s^{-1}$)", xy = (0.5, -0.28), 
        xycoords = "axes fraction", fontsize = 13,
        ha = "center"
        )

