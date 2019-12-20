CALL apoc.periodic.iterate("
CALL apoc.load.json('business_preproc.json') YIELD value RETURN value
","
MERGE (b:Business{id:value.business_id})
SET b += apoc.map.clean(value, ['name', 'latitude', 'longitude', 'review_count', 'city', 'state', 'stars', 'attributes', 'business_id','categories','address','postal_code'],[])
",{batchSize: 10000, iterateList: true});
