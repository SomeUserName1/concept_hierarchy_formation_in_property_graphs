CALL apoc.periodic.iterate("
CALL apoc.load.json('business_preproc.json') YIELD value RETURN value
","
MERGE (b:Business{id:value.business_id, name:value.name, address:value.address, postal_code:value.postal_code, latitude:value.latitude, longitude:value.longitude, review_count:value.review_count})
WITH b,value
UNWIND value.categories as category
MERGE (c:Category{name:category})
MERGE (b)-[:IN_CATEGORY]->(c)
WITH b,value
UNWIND value.attributess as attribute
MERGE (a:Attribute{name:attribute})
MERGE (b)-[:HAS_ATTRIBUTE]->(a)
WITH b,value
UNWIND value.GoodForMeal as meal
MERGE (me:Meal{kind:meal})
MERGE (b)-[:GOOD_FOR_MEAL]->(me)
WITH b,value
UNWIND value.BusinessParking as parking
MERGE (par:Parking{id:parking})
MERGE (b)-[:PARKING_POSSIBLILITY]->(par)
WITH b,value
UNWIND value.Ambience as ambience
MERGE (am:Ambience{id:ambience})
MERGE (b)-[:HAS_AMBIENCE]->(am)
WITH b,value
UNWIND value.BestNights as bestnight
MERGE (bn:BestNight{id:bestnight})
MERGE (b)-[:BEST_VISIT_NIGHT]->(bn)
WITH b,value
UNWIND value.Music as music
MERGE (m:Music{id:music})
MERGE (b)-[:PLAYS_MUSIC]->(m)
WITH b,value
UNWIND value.DietaryRestrictions as diet
MERGE (die:Diet{id:diet})
MERGE (b)-[:SUPPORTS_DIETARY_RESTRICTION]->(die)
WITH b,value
UNWIND value.HairSpecializesIn as specialization
MERGE (spec:HairSpecialization{id:specialization})
MERGE (b)-[:SPECIALIZED_IN]->(spec)
WITH b,value
MERGE (wi:WiFi{id:value.WiFi})
MERGE (b)-[:HAS_WIFI]->(wi)
MERGE (alc:Alcohol{id:value.Alcohol})
MERGE (b)-[:HAS_ALCOHOL_SORTS]->(alc)
MERGE (at:Attire{id:value.RestaurantsAttire})
MERGE (b)-[:HAS_ATTIRE]->(at)
MERGE (noi:NoiseLevel{id:value.NoiseLevel})
MERGE (b)-[:HAS_NOISE_LEVEL]->(noi)
MERGE (pr:PriceRange{id:value.RestaurantsPriceRange2})
MERGE (b)-[:HAS_PRICE_RANGE]->(pr)
MERGE (cit:City{name:value.city})
MERGE (b)-[:IN_CITY]->(cit)
MERGE (s:State{name:value.state})
MERGE (cit)-[:IN_STATE]->(s)
MERGE (sta:Stars{amount:value.stars})
MERGE (b)-[:RATING]->(sta)
MERGE (ar:AgeRestriction{id:value.AgesAllowed})
MERGE (b)-[:HAS_AGE_RESTRICTION]->(ar)
",{batchSize: 10000, iterateList: true});

