// JavaScript to be passed to `mongo` for repairing server

// Set up database instance
conn = new Mongo();
db = conn.getDB("spearmint");

// Loop collections
for (var i = 0; i < db.getCollectionNames().length; i++) {

    // Get IDs of pending jobs
    pending = db[db.getCollectionNames()[i]].find({status: 'pending'});
    for (var j = 0; j < pending.length(); j++) {
        print("Pending job with ID: " + pending[j]['id']);
    }
}
