### Uniform Format of Whole Silde Image Instance Segmentation Results Pickle File
`case_id.svs   ——(LVI-NET)——>   case_id.pickle`



```python
case_id.pickle : dict

pickle_data = {'masks':[list[tuple]],'bboxes':[list[int]],'scores':[float]}
mask : [(x,x),(x,x),(x,x),(x,x),(x,x)]
box: [x_min,y_min,x_max,y_max]
score: 0.9
```


