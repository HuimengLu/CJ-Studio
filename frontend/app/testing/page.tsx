import { redirect } from "next/navigation";

/* The style-preview editor graduated from /testing to the official /social
   page; this stub keeps old links working. */
export default function TestingRedirect() {
  redirect("/social");
}
