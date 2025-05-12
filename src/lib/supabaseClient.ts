import { createClient } from "@supabase/supabase-js";
import { Database } from "@/types/database.types";

const url = import.meta.env.VITE_SUPABASE_URL;
const key = import.meta.env.VITE_SUPABASE_KEY;

export const supabaseClient = createClient<Database>(url, key);
