CALL apoc.periodic.iterate("
CALL apoc.load.json('file:///home/someusername/Nextcloud/workspace/uni/bachelor/bachelor_project/data/business.json') YIELD value RETURN value
","

MERGE (b:Business{id:value.business_id})
SET b += apoc.map.clean(value, ['attributes','hours','business_id','categories','address','postal_code'],[])
WITH b,value.categories as categories,value.attributes as attributes

SET attribs = apoc.map.clean(attributes, ['BusinessParking', 'BestNights', 'Ambience', 'Music'], ["False", "u'no'", False, "u'none'"])
SET ambiences = apoc.map.clean(apoc.map.submap(attributes, ['Ambience'], [AddNullsForMissing]),  [], ["False", False])
SET musics = apoc.map.clean(apoc.map.submap(attributes, ['Music'], [AddNullsForMissing]), [], ["False", False])

UNWIND categories as category
MERGE (c:Category{id:category})
MERGE (b)-[:IN_CATEGORY]->(c)

UNWIND attribs as attribute
MERGE (a:Attribute{id:attribute})
MERGE (b)-[:HAS_ATTRIBUTE]->(a)

UNWIND ambiences as ambience
MERGE (am:Ambience{id:ambience})
MERGE (b)-[:HAS_AMBIENCE]->(am)

UNWIND musics as music
MERGE (m:Music{id:music})
MERGE (b)-[:PLAYS_MUSIC]->(m)


// And tons more. Does Cypher have sth that flattens jsons? 
// Else do it outside of cypher: 
// for node in businesses
//      for (jsonElem in node[attributes]):
//              if (isComposite):
//                  node[str(jsonElem)]
//                  del attributes[str(jsonElem)]
// TODO use preproced

",{batchSize: 10000, iterateList: true});

