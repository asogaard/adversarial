// JavaScript to be passed to `mongo` for repairing server
// @NOTE: The variable `experiment` must be defined by user as
// $ mongo --eval "var experiment='...'" path/to/get_pending.js

// Set up database instance
conn = new Mongo();
db = conn.getDB("spearmint");

// Get IDs of pending jobs in `experiment`
try {
    pending = db["optimisation-" + experiment + ".jobs"].find({status: 'pending'});
    for (var j = 0; j < pending.length(); j++) {
        print("Pending job with ID: " + pending[j]['id']);
    }
} catch (err) {
    if (err instanceof ReferenceError) {
        print("ERROR: The `experiment` variable was not defined by the user.");
    }
}
