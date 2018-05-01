#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from obspy import UTCDateTime
import obspy.clients.fdsn.client as fdsn

# Parameters
#arrayName = 'BH'
#staCodes = 'BH01,BH02,BH03,BH04,BH05,BH06,BH07,BH08,BH09,BH10,BH11'
#chans = 'SHE,SHN,SHZ'
#network = 'XG'

#arrayName = 'BS'
#staCodes = 'BS01,BS02,BS03,BS04,BS05,BS06,BS11,BS20,BS21,BS22,BS23,BS24,BS25,BS26,BS27'
#chans = 'SHE,SHN,SHZ'
#network = 'XU'

#arrayName = 'CL'
#staCodes = 'CL01,CL02,CL03,CL04,CL05,CL06,CL07,CL08,CL09,CL10,CL11,CL12,CL13,CL14,CL15,CL16,CL17,CL18,CL19,CL20'
#chans = 'SHE,SHN,SHZ'
#network = 'XG'

#arrayName = 'DR'
#staCodes = 'DR01,DR02,DR03,DR04,DR05,DR06,DR07,DR08,DR09,DR10,DR12'
#chans = 'SHE,SHN,SHZ'
#network = 'XG'

#arrayName = 'GC'
#staCodes = 'GC01,GC02,GC03,GC04,GC05,GC06,GC07,GC08,GC09,GC10,GC11,GC12,GC13,GC14'
#chans = 'SHE,SHN,SHZ'
#network = 'XG'

#arrayName = 'LC'
#staCodes = 'LC01,LC02,LC03,LC04,LC05,LC06,LC07,LC08,LC09,LC10,LC11,LC12,LC13,LC14'
#chans = 'SHE,SHN,SHZ'
#network = 'XG'

#arrayName = 'PA'
#staCodes = 'PA01,PA02,PA03,PA04,PA05,PA06,PA07,PA08,PA09,PA10,PA11,PA12,PA13'
#chans = 'SHE,SHN,SHZ'
#network = 'XG'

arrayName = 'TB'
staCodes = 'TB01,TB02,TB03,TB04,TB05,TB06,TB07,TB08,TB09,TB10,TB11,TB12,TB13,TB14'
chans = 'SHE,SHN,SHZ'
network = 'XG'

YY1 = '2009'
MM1 = '01'
DD1 = '01'
HH1 = '00'
mm1 = '00'
ss1 = '01'

YY2 = '2011'
MM2 = '12'
DD2 = '31'
HH2 = '23'
mm2 = '59'
ss2 = '59'

Tstart = UTCDateTime(YY1 + '-' + MM1 + '-' + DD1 + 'T' + HH1 + ':' + mm1 + \
    ':' + ss1)
Tend = UTCDateTime(YY2 + '-' + MM2 + '-' + DD2 + 'T' + HH2 + ':' + mm2 + \
    ':' + ss2)

fdsn_client = fdsn.Client('IRIS')

inventory = fdsn_client.get_stations(network=network, station=staCodes, \
    location='--', channel=chans, starttime=Tstart, endtime=Tend, \
    level='response')
inventory.write('../data/response/' + network + '_' + arrayName + '.xml', \
    format='STATIONXML') 
