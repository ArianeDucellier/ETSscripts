# Ray Tracing

This directory contains five notebooks to answer the following questions:

- AmplitudesOnGrid.ipynb: What is the average amplitude of the direct and the reflected and converted waves at the seismic station? What is the corresponding ratio between the amplitudes of the direct and reflected / converted waves? What are the phases that we can expect to see on the auto / cross correlation plots?

- TimeDifferenceArray.ipynb: At which time lags should we expect to see a peak in the auto / cross correlation signal?

- TimeLagOnGrid.ipynb: Before compute the auto / cross correlation, we are going to stack the seismograms recorded at different stations of the same array. We need to know whether we need to apply some time shift to the seismic recordings to align the time arrivals of the different phases before stacking the seismograms. For that, we need to know whether there is a significant difference between the time lags at different stations from the same array.

- TimeDifferenceStrikeDip: If we denote $t_d$ the arrival time of the direct wave, and $t_r$ the arrival time of the reflected wave, the time lag between the two phase arrivals is $tlag = t_d - t_r$. During the one-minute time window where we compute the auto / cross correlation, the displacement $dx$ of the tremor source along the plate boundary should corresponds to a time lag difference $dtlag$ shorter than a quarter of the dominant period of $T$ = 0.3 s. During an ETS event, rapid tremor streaks have been observed propagating in the up-dip and down-dip directions at velocities ranging between 30 and 110 km/h, and possibly up to 200 km/h, which corresponds to a source displacement of 0.5 to 1.8 km, and up to 3.3 km during one minute. Should we care about these tremor streaks?

- RayTracing: Computation of the time delays and the amplitude ratios for a specific array and a specific location of the tremor source.
