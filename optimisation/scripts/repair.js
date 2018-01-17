// JavaScript to be passed to `mongo` for repairing server
// @NOTE: The variable `experiment` must be defined by user as
// $ mongo --eval "var experiment='...'" path/to/repair.js

// Set up database instance
conn = new Mongo();
db = conn.getDB("spearmint");

// Remove stalled jobs in `experiment`
try {
    db["optimisation-" + experiment + ".jobs"].remove({status: 'pending'});
} catch (err) {
    if (err instanceof ReferenceError) {
        print("ERROR: The `experiment` variable was not defined by the user.");
    }
}
