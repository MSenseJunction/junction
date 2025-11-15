// seed_features_yz_from_csv.js
import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";
import dotenv from "dotenv";
dotenv.config();

import { supabase } from "./src/config/supabase.js"; // adjust path if needed

const SEED_FILE = "features_yz_seed.csv";
const TABLE_NAME = "features_yz";

async function main() {
  const csvRaw = fs.readFileSync(path.resolve(SEED_FILE), "utf8");

  const records = parse(csvRaw, {
    columns: true,
    skip_empty_lines: true,
  });

  console.log(`Read ${records.length} rows from ${SEED_FILE}`);

  // Optional: clear existing rows for this user first
  const userId = "YZMM";
  const { error: delError } = await supabase
    .from(TABLE_NAME)
    .delete()
    .eq("user_id", userId);

  if (delError) {
    console.error("Error clearing old rows (you can ignore if table is empty):", delError.message);
  } else {
    console.log(`Cleared existing rows for user ${userId} in ${TABLE_NAME}`);
  }

  const rowsForDb = records.map((r) => ({
    user_id: r.user_id,
    timestamp: r.timestamp,
    workload: parseFloat(r.workload),
    stress: parseFloat(r.stress),
    hrv: parseFloat(r.hrv),
  }));

  const { error: insertError } = await supabase
    .from(TABLE_NAME)
    .insert(rowsForDb);

  if (insertError) {
    console.error("Insert error:", insertError);
    process.exit(1);
  }

  console.log(`Successfully inserted ${rowsForDb.length} rows into ${TABLE_NAME}`);
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
