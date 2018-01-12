// JavaScript to be passed to `mongo` for repairing server

// Set up database instance
conn = new Mongo();
db = conn.getDB("spearmint");

// Loop collections
for (var i = 0; i < db.getCollectionNames().length; i++) {

  // Remove stalled jobs
  db[db.getCollectionNames()[i]].remove({status: 'pending'})
}
