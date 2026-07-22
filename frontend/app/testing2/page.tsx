import { redirect } from "next/navigation";

/* The gpt-image-2 pipeline graduated from /testing2 to the root New Listing
   page; this stub keeps old links working. */
export default function Testing2Redirect() {
  redirect("/");
}
