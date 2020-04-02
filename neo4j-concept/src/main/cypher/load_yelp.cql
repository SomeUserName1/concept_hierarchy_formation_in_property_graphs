CALL apoc.periodic.iterate("
CALL apoc.load.json('business_preproc.json') YIELD value RETURN value
","
MERGE (b:Business{id:value.business_id, name:value.name, address:value.address, postal_code:value.postal_code, latitude:value.latitude, longitude:value.longitude, review_count:value.review_count})

FOREACH( ignore IN CASE WHEN value.RestaurantsAttire IS NULL THEN [] ELSE [1] END |
    FOREACH(category IN value.categories |
        MERGE (c:Category{name:category})
        MERGE (b)-[:IN_CATEGORY]->(c)
    )
)

FOREACH( ignore IN CASE WHEN value.attributes IS NULL THEN [] ELSE [1] END |
    FOREACH(attribute IN value.attributes |
        MERGE (a:Attribute{name:attribute})
        MERGE (b)-[:HAS_ATTRIBUTE]->(a)
    )
)

FOREACH( ignore IN CASE WHEN value.GoodForMeal IS NULL THEN [] ELSE [1] END |
    FOREACH(meal IN value.GoodForMeal |
        MERGE (me:Meal{kind:meal})
        MERGE (b)-[:GOOD_FOR_MEAL]->(me)
    )
)

FOREACH( ignore IN CASE WHEN value.BusinessParking IS NULL THEN [] ELSE [1] END |
    FOREACH(parking IN value.BusinessParking |
        MERGE (par:Parking{id:parking})
        MERGE (b)-[:PARKING_POSSIBLILITY]->(par)
    )
)

FOREACH( ignore IN CASE WHEN value.Ambience IS NULL THEN [] ELSE [1] END |
    FOREACH(ambience IN value.Ambience |
        MERGE (am:Ambience{id:ambience})
        MERGE (b)-[:HAS_AMBIENCE]->(am)
    )
)

FOREACH( ignore IN CASE WHEN value.BestNights IS NULL THEN [] ELSE [1] END |
    FOREACH(bestnight IN value.BestNights |
        MERGE (bn:BestNight{id:bestnight})
        MERGE (b)-[:BEST_VISIT_NIGHT]->(bn)
    )
)

FOREACH( ignore IN CASE WHEN value.Music IS NULL THEN [] ELSE [1] END |
    FOREACH(music IN value.Music |
        MERGE (m:Music{id:music})
        MERGE (b)-[:PLAYS_MUSIC]->(m)
    )
)

FOREACH( ignore IN CASE WHEN value.DietaryRestrictions IS NULL THEN [] ELSE [1] END |
    FOREACH(diet IN value.DietaryRestrictions |
        MERGE (die:Diet{id:diet})
        MERGE (b)-[:SUPPORTS_DIETARY_RESTRICTION]->(die)
    )
)

FOREACH( ignore IN CASE WHEN value.HairSpecializesIn IS NULL THEN [] ELSE [1] END |
    FOREACH(specialization IN value.HairSpecializesIn |
        MERGE (spec:HairSpecialization{id:specialization})
        MERGE (b)-[:SPECIALIZED_IN]->(spec)
    )
)

FOREACH( ignore IN CASE WHEN value.WiFi IS NULL THEN [] ELSE [1] END |
    MERGE (wi:WiFi{id:value.WiFi})
    MERGE (b)-[:HAS_WIFI]->(wi)
)

FOREACH( ignore IN CASE WHEN value.Alcohol IS NULL THEN [] ELSE [1] END |
    MERGE (alc:Alcohol{id:value.Alcohol})
    MERGE (b)-[:HAS_ALCOHOL_SORTS]->(alc)
)

FOREACH( ignore IN CASE WHEN value.RestaurantsAttire IS NULL THEN [] ELSE [1] END |
    MERGE (at:Attire{id:value.RestaurantsAttire})
    MERGE (b)-[:HAS_ATTIRE]->(at)
)

FOREACH( ignore IN CASE WHEN value.NoiseLevel IS NULL THEN [] ELSE [1] END |
    MERGE (noi:NoiseLevel{id:value.NoiseLevel})
    MERGE (b)-[:HAS_NOISE_LEVEL]->(noi)
)

FOREACH( ignore IN CASE WHEN value.RestaurantsPriceRange2 IS NULL THEN [] ELSE [1] END |
    MERGE (pr:PriceRange{id:value.RestaurantsPriceRange2})
    MERGE (b)-[:HAS_PRICE_RANGE]->(pr)
)

FOREACH( ignore IN CASE WHEN value.city IS NULL OR value.state IS NULL THEN [] ELSE [1] END |
    MERGE (cit:City{name:value.city})
    MERGE (b)-[:IN_CITY]->(cit)
    MERGE (s:State{name:value.state})
    MERGE (cit)-[:IN_STATE]->(s)
)

FOREACH( ignore IN CASE WHEN value.stars IS NULL THEN [] ELSE [1] END |
    MERGE (sta:Stars{amount:value.stars})
    MERGE (b)-[:RATING]->(sta)
)


FOREACH( ignore IN CASE WHEN value.AgesAllowed IS NULL THEN [] ELSE [1] END |
    MERGE (ar:AgeRestriction{id:value.AgesAllowed})
    MERGE (b)-[:HAS_AGE_RESTRICTION]->(ar)
)
",{batchSize: 128, iterateList: true});

