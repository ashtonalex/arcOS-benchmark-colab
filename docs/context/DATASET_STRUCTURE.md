======================================================================
DATA STRUCTURE ANALYSIS — First 3 examples
======================================================================

──────────────────────────────────────────────────────────────────────
EXAMPLE 0: WebQTrn-0
──────────────────────────────────────────────────────────────────────
  Question:  what is the name of justin bieber brother
  Answer:    ['Jaxon Bieber']
  q_entity:  ['Justin Bieber']  (type: list)
  a_entity:  ['Jaxon Bieber']  (type: list)
  Graph:     9088 triples

  Unique subjects:  981
  Unique objects:   1723
  Unique entities:  1723
  Unique relations: 335

  First 10 triples:
    [0] 'P!nk'  --(freebase.valuenotation.is_reviewed)-->  'Gender'
    [1] '1Club.FM: Power'  --(broadcast.content.artist)-->  'P!nk'
    [2] 'Somebody to Love'  --(music.recording.contributions)-->  'm.0rqp4h0'
    [3] 'Rudolph Valentino'  --(freebase.valuenotation.is_reviewed)-->  'Place of birth'
    [4] 'Ice Cube'  --(broadcast.artist.content)-->  '.977 The Hits Channel'
    [5] 'Colbie Caillat'  --(broadcast.artist.content)-->  'Hot Wired Radio'
    [6] 'Stephen Melton'  --(people.person.nationality)-->  'United States of America'
    [7] 'Record producer'  --(music.performance_role.regular_performances)-->  'm.012m1vf1'
    [8] 'Justin Bieber'  --(award.award_winner.awards_won)-->  'm.0yrkc0l'
    [9] '1.FM Top 40'  --(broadcast.content.artist)-->  'Geri Halliwell'

  Last 5 triples:
    [9083] 'Justin Bieber'  --(music.artist.concert_tours)-->  'Believe Tour'
    [9084] 'Janet Jackson'  --(people.person.languages)-->  'English Language'
    [9085] 'Live My Life'  --(music.recording.tracks)-->  'Live My Life'
    [9086] 'Don Henley'  --(people.person.profession)-->  'Actor'
    [9087] 'Parents'  --(rdf-schema#range)-->  'Person'

  q_entity in graph nodes: 'Justin Bieber' -> YES  
  a_entity in graph nodes: 'Jaxon Bieber' -> YES  

  Top 5 relations:
     902x  freebase.valuenotation.is_reviewed
     771x  broadcast.content.artist
     743x  music.artist.genre
     631x  broadcast.artist.content
     616x  people.person.profession

  Sample entity names (first 15 alphabetically):
    '#Musik.Main on RauteMusik.FM'
    '#ThatPower (feat. Justin Bieber)'
    '#Thatpower'
    '#thatPOWER'
    '#thatPower'
    '#thatPower (remix)'
    '#thatpower'
    '.977 The Hits Channel'
    "00's"
    '1.FM Top 40'
    '1.FM Top 40 - 32kbps Stream'
    '112'
    '181-beat'
    '181-party'
    '181-rnb'

──────────────────────────────────────────────────────────────────────
EXAMPLE 1: WebQTrn-1
──────────────────────────────────────────────────────────────────────
  Question:  what character did natalie portman play in star wars
  Answer:    ['Padmé Amidala']
  q_entity:  ['Natalie Portman']  (type: list)
  a_entity:  ['Padmé Amidala']  (type: list)
  Graph:     4135 triples

  Unique subjects:  930
  Unique objects:   1253
  Unique entities:  1253
  Unique relations: 267

  First 10 triples:
    [0] 'm.0t4tf9z'  --(film.film_crew_gig.film)-->  'Hesher'
    [1] 'Jonathan Glickman'  --(film.producer.films_executive_produced)-->  'No Strings Attached'
    [2] 'Anywhere but Here'  --(film.film.language)-->  'English Language'
    [3] 'Israeli American'  --(people.ethnicity.people)-->  'Shiri Appleby'
    [4] 'm.0g4xwxr'  --(film.film_crew_gig.film)-->  'Love and Other Impossible Pursuits'
    [5] 'Veganism'  --(religion.religious_practice.practice_of)-->  'Jainism'
    [6] 'm.0w9tk0s'  --(tv.tv_guest_personal_appearance.person)-->  'Natalie Portman'
    [7] 'm.0nfwmy8'  --(award.award_nomination.award)-->  'SFX Award for Best Actress'
    [8] 'Hesher'  --(film.film.directed_by)-->  'Spencer Susser'
    [9] 'Zac Posen'  --(base.popstra.company.fashion_choice)-->  'm.077jny5'

  Last 5 triples:
    [4130] 'm.090dzg7'  --(award.award_nomination.award)-->  'Golden Globe Award for Best Supporting Actress – Motion Picture'
    [4131] 'No Strings Attached'  --(film.film.production_companies)-->  'Paramount Pictures'
    [4132] 'Natalie Portman'  --(film.director.film)-->  'A Tale of Love and Darkness'
    [4133] 'Natalie Portman'  --(people.measured_person.measurements)-->  'm.012sr3qf'
    [4134] 'Parents'  --(rdf-schema#range)-->  'Person'

  q_entity in graph nodes: 'Natalie Portman' -> YES  
  a_entity in graph nodes: 'Padmé Amidala' -> YES  

  Top 5 relations:
     325x  freebase.valuenotation.is_reviewed
     208x  film.film.starring
     207x  film.performance.film
     138x  people.person.profession
     134x  common.topic.notable_types

  Sample entity names (first 15 alphabetically):
    "10th Critics' Choice Awards"
    '10th Satellite Awards'
    '14A (Canada)'
    '15th Satellite Awards'
    "16th Critics' Choice Awards"
    '17th Screen Actors Guild Awards'
    '2000 Teen Choice Awards'
    '2002 Teen Choice Awards'
    '2005 MTV Movie Awards'
    '2005 Teen Choice Awards'
    '2006 Teen Choice Awards'
    '2009 Teen Choice Awards'
    '2009 Toronto International Film Festival'
    '2010 Sundance Film Festival'
    '2011 MTV Movie Awards'

──────────────────────────────────────────────────────────────────────
EXAMPLE 2: WebQTrn-3
──────────────────────────────────────────────────────────────────────
  Question:  what country is the grand bahama island in
  Answer:    ['Bahamas']
  q_entity:  ['Grand Bahama']  (type: list)
  a_entity:  ['Bahamas']  (type: list)
  Graph:     2174 triples

  Unique subjects:  411
  Unique objects:   1286
  Unique entities:  1286
  Unique relations: 168

  First 10 triples:
    [0] 'Acklins'  --(location.administrative_division.first_level_division_of)-->  'Bahamas'
    [1] 'Geographical Feature'  --(freebase.type_profile.strict_included_types)-->  'Location'
    [2] 'Hurricane Edith'  --(meteorology.tropical_cyclone.affected_areas)-->  'Bahamas'
    [3] 'Bahamas'  --(meteorology.cyclone_affected_area.cyclones)-->  '1933 Treasure Coast hurricane'
    [4] 'Bahamas'  --(location.statistical_region.part_time_employment_percent)-->  'g.1hhc4bpw0'
    [5] 'm.04kc4ps'  --(organization.organization_membership.member)-->  'Bahamas'
    [6] 'Bahamas'  --(location.statistical_region.gender_balance_members_of_parliament)-->  'g.1hhc3s6q7'
    [7] 'Politician'  --(freebase.type_profile.equivalent_topic)-->  'Politician'
    [8] 'Bahamas'  --(location.statistical_region.gni_in_ppp_dollars)-->  'g.1245_mkq0'
    [9] 'Bahamas'  --(location.statistical_region.co2_emissions_per_capita)-->  'g.12460kjms'

  Last 5 triples:
    [2169] 'Bahamas'  --(location.statistical_region.co2_emissions_per_capita)-->  'm.0nf6myd'
    [2170] 'Hurricane Rita'  --(meteorology.tropical_cyclone.affected_areas)-->  'Bahamas'
    [2171] 'Alice Town'  --(common.topic.notable_types)-->  'Location'
    [2172] 'Bahamas'  --(location.statistical_region.internet_users_percent_population)-->  'g.1245zfnj0'
    [2173] 'Cat Island, Bahamas'  --(location.administrative_division.first_level_division_of)-->  'Bahamas'

  q_entity in graph nodes: 'Grand Bahama' -> YES  
  a_entity in graph nodes: 'Bahamas' -> YES  

  Top 5 relations:
     121x  location.location.containedby
     120x  location.location.contains
     100x  location.statistical_region.co2_emissions_per_capita
      86x  meteorology.tropical_cyclone.affected_areas
      86x  meteorology.cyclone_affected_area.cyclones

  Sample entity names (first 15 alphabetically):
    '1891 Martinique hurricane'
    '1899 San Ciriaco hurricane'
    '1900 Galveston hurricane'
    '1919 Florida Keys hurricane'
    '1924 Cuba hurricane'
    '1926 Miami hurricane'
    '1926 Nassau hurricane'
    '1928 Okeechobee hurricane'
    '1929 Bahamas hurricane'
    '1932 Bahamas hurricane'
    '1932 Cuba hurricane'
    '1933 Treasure Coast hurricane'
    '1935 Labor Day hurricane'
    '1935 Yankee hurricane'
    '1938 New England hurricane'

======================================================================
CROSS-EXAMPLE SUMMARY
======================================================================

Entity overlap between examples:
  Ex0 ∩ Ex1: 41 shared entities
  Ex0 ∩ Ex2: 6 shared entities
  Ex1 ∩ Ex2: 6 shared entities
  Sample shared (0∩1): ['2011 MTV Movie Awards', '2011 Teen Choice Awards', 'Actor', 'Artist', 'Award-Winning Work']
